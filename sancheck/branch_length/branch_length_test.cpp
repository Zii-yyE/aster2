#include "../../src/caster.hpp"
#include "../../src/branch_length.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

using TestColor = caster::Color<caster::StepwiseColorDefaultAttributes<unsigned char, unsigned short>>;

static std::size_t canonicalClass(std::array<std::size_t, 4> nucleotide) {
	static constexpr std::array<std::array<unsigned char, 4>, 15> classes = {{
		{{0, 0, 0, 0}}, {{0, 0, 0, 1}}, {{0, 0, 1, 0}}, {{0, 0, 1, 1}},
		{{0, 0, 1, 2}}, {{0, 1, 0, 0}}, {{0, 1, 0, 1}}, {{0, 1, 0, 2}},
		{{0, 1, 1, 0}}, {{0, 1, 1, 1}}, {{0, 1, 1, 2}}, {{0, 1, 2, 0}},
		{{0, 1, 2, 1}}, {{0, 1, 2, 2}}, {{0, 1, 2, 3}}
	}};
	std::array<signed char, 4> mapping = {{-1, -1, -1, -1}};
	std::array<unsigned char, 4> renamed{};
	unsigned char next = 0;
	for (std::size_t i = 0; i < 4; ++i) {
		if (mapping[nucleotide[i]] == -1) mapping[nucleotide[i]] = next++;
		renamed[i] = mapping[nucleotide[i]];
	}
	for (std::size_t i = 0; i < classes.size(); ++i) if (renamed == classes[i]) return i;
	std::abort();
}

static void testPooledCounts() {
	TestColor::SharedConstData data;
	TestColor::SharedConstData::Element element;
	element.iGenomePosBegin = 0;
	element.nPos = 1;
	element.iAllGenomePosBegin = 0;
	element.nAllPos = 6;
	element.taxon2row.resize(8);
	element.cnts.resize(8, std::vector<std::array<unsigned char, 4>>(1));
	element.allCnts.resize(8, std::vector<std::array<unsigned char, 4>>(6));
	for (std::size_t i = 0; i < 8; ++i) element.taxon2row[i] = i;

	// Invariant, singleton, all-different, repeated-state, missing, and
	// multi-sample-count columns.
	std::array<std::array<int, 8>, 6> states = {{
		{{0, 0, 0, 0, 0, 0, 0, 0}},
		{{0, 0, 0, 0, 0, 0, 0, 1}},
		{{0, 1, 1, 1, 2, 2, 3, 3}},
		{{0, 1, 0, 1, 2, 3, 2, 3}},
		{{0, -1, 1, -1, 2, -1, 3, -1}},
		{{0, 0, 1, 1, 2, 2, 3, 3}}
	}};
	for (std::size_t site = 0; site < states.size(); ++site) {
		for (std::size_t taxon = 0; taxon < 8; ++taxon) {
			int state = states[site][taxon];
			if (state >= 0) element.allCnts[taxon][site][state]++;
		}
	}
	element.allCnts[0][5][0]++; // a second A observation for taxon 0
	element.cnts[0][0][0] = 1;
	element.cnts[2][0][0] = 1;
	element.cnts[4][0][0] = 1;
	element.cnts[6][0][0] = 1;
	data.elements.push_back(element);
	data.nGenomePos = 1;
	data.nAllGenomePos = 6;

	TestColor color(data, TestColor::SiteView::ALL_SITES);
	for (std::size_t taxon = 0; taxon < 8; ++taxon)
		color.elementSetTaxonColor(0, taxon, taxon / 2);
	TestColor::PatternCounts pooled{};
	color.elementAccumulateQuartetPatternCounts(0, {{0, 1, 2, 3}}, pooled);

	TestColor::PatternCounts brute{};
	std::array<long double, 256> pooledRaw{};
	std::array<long double, 256> bruteRaw{};
	for (std::size_t site = 0; site < 6; ++site)
	for (std::size_t a = 0; a < 4; ++a) for (std::size_t b = 0; b < 4; ++b)
	for (std::size_t c = 0; c < 4; ++c) for (std::size_t d = 0; d < 4; ++d) {
		const std::size_t raw = (a << 6) | (b << 4) | (c << 2) | d;
		long double groupA = element.allCnts[0][site][a] + element.allCnts[1][site][a];
		long double groupB = element.allCnts[2][site][b] + element.allCnts[3][site][b];
		long double groupC = element.allCnts[4][site][c] + element.allCnts[5][site][c];
		long double groupD = element.allCnts[6][site][d] + element.allCnts[7][site][d];
		pooledRaw[raw] += groupA * groupB * groupC * groupD;

	for (std::size_t ta = 0; ta < 2; ++ta) for (std::size_t tb = 2; tb < 4; ++tb)
	for (std::size_t tc = 4; tc < 6; ++tc) for (std::size_t td = 6; td < 8; ++td) {
		long double contribution =
			(long double)element.allCnts[ta][site][a] * element.allCnts[tb][site][b] *
			element.allCnts[tc][site][c] * element.allCnts[td][site][d];
		bruteRaw[raw] += contribution;
		brute[canonicalClass({{a, b, c, d}})] += contribution;
	}
	}
	assert(pooledRaw == bruteRaw);
	assert(pooled == brute);
	assert(pooled[0] > 0);  // invariant sites were retained
	assert(pooled[1] > 0);  // singleton class was retained
}

static void testProbabilitiesAndOptimizer() {
	constexpr std::array<int, 15> multiplicity = {
		4, 12, 12, 12, 24, 12, 12, 24, 12, 12, 24, 24, 24, 24, 24
	};
	auto probabilities = branch_length::jc69_msc::probabilities(
		0.1L, 0.12L, 0.14L, 0.16L, 0.08L, 0.04L);
	long double normalization = 0;
	std::array<long double, 15> expectedCounts{};
	for (std::size_t i = 0; i < probabilities.size(); ++i) {
		assert(probabilities[i] > 0);
		normalization += multiplicity[i] * probabilities[i];
		expectedCounts[i] = 100000 * multiplicity[i] * probabilities[i];
	}
	assert(std::abs(normalization - 1) < 1e-12L);
	auto fit = branch_length::Estimator<decltype(expectedCounts)>::fit(expectedCounts);
	assert(fit.success);
	assert(std::abs(fit.focalSubstitution - 0.08L) < 5e-3L);
	assert(std::abs(fit.theta - 0.04L) < 5e-3L);
}

static void testPositionalPermutationSymmetry() {
	const long double a = 0.031L;
	const long double b = 0.047L;
	const long double c = 0.059L;
	const long double d = 0.071L;
	const long double t = 0.083L;
	const long double theta = 0.097L;
	const auto p = branch_length::jc69_msc::probabilities(a, b, c, d, t, theta);
	const auto pSwapAB = branch_length::jc69_msc::probabilities(b, a, c, d, t, theta);
	const auto pSwapCherries = branch_length::jc69_msc::probabilities(c, d, a, b, t, theta);

	for (std::size_t xA = 0; xA < 4; ++xA)
	for (std::size_t xB = 0; xB < 4; ++xB)
	for (std::size_t xC = 0; xC < 4; ++xC)
	for (std::size_t xD = 0; xD < 4; ++xD) {
		const auto original = canonicalClass({{xA, xB, xC, xD}});
		const auto swapAB = canonicalClass({{xB, xA, xC, xD}});
		const auto swapCherries = canonicalClass({{xC, xD, xA, xB}});
		assert(std::abs(p[original] - pSwapAB[swapAB]) < 2e-12L);
		assert(std::abs(p[original] - pSwapCherries[swapCherries]) < 2e-12L);
	}
}

int main() {
	testPooledCounts();
	testProbabilitiesAndOptimizer();
	testPositionalPermutationSymmetry();
	std::cout << "branch-length tests passed\n";
}
