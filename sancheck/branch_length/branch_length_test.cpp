#include "../../src/caster.hpp"
#include "../../src/branch_length.hpp"
#include "../../src/terminal_branch_length.hpp"

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

static std::size_t canonicalTripletClass(
	std::array<std::size_t, 3> nucleotide
) {
	static constexpr std::array<std::array<unsigned char, 3>, 5> classes = {{
		{{0, 0, 0}}, {{0, 0, 1}}, {{0, 1, 0}},
		{{0, 1, 1}}, {{0, 1, 2}}
	}};
	std::array<signed char, 4> mapping = {{-1, -1, -1, -1}};
	std::array<unsigned char, 3> renamed{};
	unsigned char next = 0;
	for (std::size_t i = 0; i < 3; ++i) {
		if (mapping[nucleotide[i]] == -1) mapping[nucleotide[i]] = next++;
		renamed[i] = mapping[nucleotide[i]];
	}
	for (std::size_t i = 0; i < classes.size(); ++i)
		if (renamed == classes[i]) return i;
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

	TestColor::TripletPatternCounts pooledTriplet{};
	color.elementAccumulateTripletPatternCounts(
		0, {{0, 1, 2}}, pooledTriplet
	);
	TestColor::TripletPatternCounts bruteTriplet{};
	std::array<long double, 64> pooledTripletRaw{};
	std::array<long double, 64> bruteTripletRaw{};
	for (std::size_t site = 0; site < 6; ++site)
	for (std::size_t a = 0; a < 4; ++a)
	for (std::size_t b = 0; b < 4; ++b)
	for (std::size_t c = 0; c < 4; ++c) {
		const std::size_t raw = (a << 4) | (b << 2) | c;
		long double groupA =
			element.allCnts[0][site][a] + element.allCnts[1][site][a];
		long double groupB =
			element.allCnts[2][site][b] + element.allCnts[3][site][b];
		long double groupC =
			element.allCnts[4][site][c] + element.allCnts[5][site][c];
		pooledTripletRaw[raw] += groupA * groupB * groupC;

		for (std::size_t ta = 0; ta < 2; ++ta)
		for (std::size_t tb = 2; tb < 4; ++tb)
		for (std::size_t tc = 4; tc < 6; ++tc) {
			long double contribution =
				(long double)element.allCnts[ta][site][a] *
				element.allCnts[tb][site][b] *
				element.allCnts[tc][site][c];
			bruteTripletRaw[raw] += contribution;
			bruteTriplet[canonicalTripletClass({{a, b, c}})] +=
				contribution;
		}
	}
	assert(pooledTripletRaw == bruteTripletRaw);
	assert(pooledTriplet == bruteTriplet);
	assert(pooledTriplet[0] > 0);
	assert(pooledTriplet[1] > 0);
}

static void testTerminalTraversalCounts() {
	using Tree = common::AnnotatedBinaryTree;
	TestColor::SharedConstData data;
	TestColor::SharedConstData::Element element;
	element.iAllGenomePosBegin = 0;
	element.nAllPos = 6;
	element.taxon2row.resize(4);
	element.allCnts.resize(
		4, std::vector<std::array<unsigned char, 4>>(6)
	);
	for (std::size_t i = 0; i < 4; ++i) element.taxon2row[i] = i;
	std::array<std::array<int, 4>, 6> states = {{
		{{0, 0, 0, 0}},
		{{0, 0, 0, 1}},
		{{0, 1, 2, 3}},
		{{0, 1, 0, 1}},
		{{0, -1, 2, 3}},
		{{0, 1, 2, 3}}
	}};
	for (std::size_t site = 0; site < states.size(); ++site)
	for (std::size_t taxon = 0; taxon < 4; ++taxon) {
		int state = states[site][taxon];
		if (state >= 0) element.allCnts[taxon][site][state]++;
	}
	element.allCnts[0][5][0]++;
	data.elements.push_back(element);
	data.nAllGenomePos = 6;

	Tree tree;
	Tree::Node* a = tree.emplaceRoot(
		Tree::LEAF_ID, (std::size_t)0
	);
	a->emplaceAbove(Tree::LEAF_ID, (std::size_t)1);
	tree.root()->emplaceAbove(Tree::LEAF_ID, (std::size_t)2);
	Tree::Node* outgroup =
		tree.root()->emplaceAbove(Tree::LEAF_ID, (std::size_t)3);

	TestColor color(data, TestColor::SiteView::ALL_SITES);
	using Traversal =
		terminal_branch_length::PooledTripletTraversal<TestColor>;
	Traversal::ThreadPool threadPool(1, 0, data.nElements());
	Traversal traversal(color, threadPool, tree, outgroup);
	auto pooledByLeaf = traversal.count();
	assert(pooledByLeaf.size() == 4);

	const std::array<std::array<std::vector<std::size_t>, 3>, 4> groups = {{
		{{{{0}}, {{1}}, {{2, 3}}}},
		{{{{1}}, {{0}}, {{2, 3}}}},
		{{{{2}}, {{0, 1}}, {{3}}}},
		{{{{0, 1}}, {{2}}, {{3}}}}
	}};
	std::array<TestColor::TripletPatternCounts, 4> brute{};
	for (std::size_t focal = 0; focal < groups.size(); ++focal)
	for (std::size_t site = 0; site < states.size(); ++site)
	for (std::size_t x = 0; x < 4; ++x)
	for (std::size_t y = 0; y < 4; ++y)
	for (std::size_t z = 0; z < 4; ++z) {
		long double groupX = 0, groupY = 0, groupZ = 0;
		for (std::size_t taxon : groups[focal][0])
			groupX += element.allCnts[taxon][site][x];
		for (std::size_t taxon : groups[focal][1])
			groupY += element.allCnts[taxon][site][y];
		for (std::size_t taxon : groups[focal][2])
			groupZ += element.allCnts[taxon][site][z];
		brute[focal][canonicalTripletClass({{x, y, z}})] +=
			groupX * groupY * groupZ;
	}
	for (auto const& [leaf, counts] : pooledByLeaf) {
		std::size_t focal = leaf->get<std::size_t>(Tree::LEAF_ID);
		assert(counts == brute[focal]);
	}
}

static void testProbabilitiesAndOptimizer() {
	constexpr std::array<int, 15> multiplicity = {
		4, 12, 12, 12, 24, 12, 12, 24, 12, 12, 24, 24, 24, 24, 24
	};
	auto unbalanced = branch_length::jc69_msc::unbalanced(
		0.1L, 0.2L, 0.3L, 0.05L);
	auto balanced = branch_length::jc69_msc::balanced(
		0.1L, 0.15L, 0.3L, 0.05L);
	constexpr std::array<long double, 15> pythonUnbalanced = {{
		0.09697429164261531L, 0.015846430736900327L,
		0.00841624532035207L, 0.00486956158346461L,
		0.002127472693899086L, 0.0043983387594188515L,
		0.0010549684206337423L, 0.0007913167299622781L,
		0.0010549684206337423L, 0.0043983387594188515L,
		0.0007913167299622789L, 0.0004834429716501758L,
		0.00048344297165017655L, 0.0005850996707013489L,
		0.00022276695799432466L
	}};
	constexpr std::array<long double, 15> pythonBalanced = {{
		0.10091452415975577L, 0.007102834481305252L,
		0.007102834481305252L, 0.013761087009090579L,
		0.002053508108883654L, 0.004983284088776595L,
		0.0006184507986972812L, 0.00040922911793954024L,
		0.0006184507986972812L, 0.004983284088776596L,
		0.00040922911793954013L, 0.00040922911793954024L,
		0.00040922911793954013L, 0.001372034312085861L,
		0.0002000075406552963L
	}};
	long double unbalancedNormalization = 0;
	long double balancedNormalization = 0;
	std::array<long double, 15> expectedCounts{};
	for (std::size_t i = 0; i < unbalanced.size(); ++i) {
		assert(unbalanced[i] > 0);
		assert(balanced[i] > 0);
		assert(std::abs(unbalanced[i] - pythonUnbalanced[i]) < 2e-15L);
		assert(std::abs(balanced[i] - pythonBalanced[i]) < 2e-15L);
		unbalancedNormalization += multiplicity[i] * unbalanced[i];
		balancedNormalization += multiplicity[i] * balanced[i];
		expectedCounts[i] =
			1000000 * multiplicity[i] * unbalanced[i];
	}
	assert(std::abs(unbalancedNormalization - 1) < 1e-12L);
	assert(std::abs(balancedNormalization - 1) < 1e-12L);

	constexpr std::array<long double, 15> benchmarkCounts = {{
		996435, 1776, 593, 192, 0,
		442, 57, 0, 53, 452,
		0, 0, 0, 0, 0
	}};
	long double fixedLogLikelihood = 0;
	for (std::size_t i = 0; i < unbalanced.size(); ++i)
		if (benchmarkCounts[i])
			fixedLogLikelihood +=
				benchmarkCounts[i] * std::log(unbalanced[i]);
	assert(
		std::abs(fixedLogLikelihood + 2341813.05389320L) < 1e-6L
	);

	auto fit = branch_length::Estimator<decltype(expectedCounts)>::fit(
		expectedCounts, branch_length::RootedQuartetShape::UNBALANCED);
	assert(fit.success);
	assert(std::abs(fit.focalSubstitution - 0.1L) < 5e-3L);
	assert(std::abs(fit.theta - 0.05L) < 5e-3L);
}

static void testTripletProbabilitiesAndOptimizer() {
	constexpr std::array<int, 5> multiplicity = {{4, 12, 12, 12, 24}};
	constexpr std::array<long double, 5> pythonTriplet = {{
		0.144513583853316291L,
		0.017540752291614852L,
		0.0070359406399771500L,
		0.0070359406399771516L,
		0.00177475257199602591L
	}};
	auto triplet =
		branch_length::jc69_msc::triplet(0.1L, 0.2L, 0.05L);
	auto quartet1 =
		branch_length::jc69_msc::unbalanced(
			0.1L, 0.2L, 0.3L, 0.05L
		);
	auto quartet2 =
		branch_length::jc69_msc::unbalanced(
			0.1L, 0.2L, 0.9L, 0.05L
		);
	auto marginalize = [](auto const& quartet) {
		return std::array<long double, 5> {{
			quartet[0] + 3 * quartet[1],
			quartet[2] + quartet[3] + 2 * quartet[4],
			quartet[5] + quartet[6] + 2 * quartet[7],
			quartet[8] + quartet[9] + 2 * quartet[10],
			quartet[11] + quartet[12] + quartet[13] + quartet[14]
		}};
	};
	auto marginal1 = marginalize(quartet1);
	auto marginal2 = marginalize(quartet2);

	long double normalization = 0;
	std::array<long double, 5> expectedCounts{};
	for (std::size_t i = 0; i < triplet.size(); ++i) {
		assert(triplet[i] > 0);
		assert(std::abs(triplet[i] - pythonTriplet[i]) < 2e-15L);
		assert(std::abs(triplet[i] - marginal1[i]) < 2e-15L);
		assert(std::abs(triplet[i] - marginal2[i]) < 2e-15L);
		normalization += multiplicity[i] * triplet[i];
		expectedCounts[i] =
			1000000 * multiplicity[i] * triplet[i];
	}
	assert(std::abs(normalization - 1) < 1e-12L);
	assert(std::abs(triplet[1] - triplet[2]) > 1e-3L);
	constexpr std::array<std::array<long double, 3>, 3> points = {{
		{{0.00025L, 0.00035L, 0.0005L}},
		{{0.03L, 0.19L, 0.08L}},
		{{0.4L, 0.9L, 0.7L}}
	}};
	for (auto point : points) {
		auto probabilities =
			branch_length::jc69_msc::triplet(
				point[0], point[1], point[2]
			);
		long double sum = 0;
		for (std::size_t i = 0; i < probabilities.size(); ++i) {
			assert(probabilities[i] > 0);
			sum += multiplicity[i] * probabilities[i];
		}
		assert(std::abs(sum - 1) < 2e-11L);
	}

	constexpr std::array<long double, 5> likelihoodCounts = {{
		1000, 120, 80, 70, 30
	}};
	long double fixedLogLikelihood =
		terminal_branch_length::Estimator<
			decltype(likelihoodCounts)
		>::logLikelihood(
			likelihoodCounts, 0.1L, 0.2L, 0.05L
		);
	assert(
		std::abs(fixedLogLikelihood + 3353.1005894591267L) < 1e-9L
	);

	auto fixedThetaFit =
		terminal_branch_length::Estimator<
			decltype(expectedCounts)
		>::fit(expectedCounts, 0.05L);
	assert(fixedThetaFit.success);
	assert(std::abs(fixedThetaFit.speciesAges[0] - 0.1L) < 5e-3L);
	assert(std::abs(fixedThetaFit.speciesAges[1] - 0.2L) < 5e-3L);
	assert(std::abs(fixedThetaFit.theta - 0.05L) < 1e-15L);

	auto jointFit =
		terminal_branch_length::Estimator<
			decltype(expectedCounts)
		>::fit(expectedCounts);
	assert(jointFit.success);
	assert(std::abs(jointFit.speciesAges[0] - 0.1L) < 5e-3L);
	assert(std::abs(jointFit.speciesAges[1] - 0.2L) < 5e-3L);
	assert(std::abs(jointFit.theta - 0.05L) < 5e-3L);
}

static void testPythonBenchmarkFit() {
	// Exact pooled A,B,C,D counts from the generated 1 Mb JC69 quartet:
	// /simulation/jc_1mb_n4/simulated_alignment_1mb.fasta.
	// The Python reference, with the same 200 iterations and five restarts,
	// estimates focal SU=0.0001390646 and theta=0.0005450746.
	std::array<long double, 15> counts = {{
		996435, 1776, 593, 192, 0,
		442, 57, 0, 53, 452,
		0, 0, 0, 0, 0
	}};
	auto fit = branch_length::Estimator<decltype(counts)>::fit(
		counts, branch_length::RootedQuartetShape::UNBALANCED);
	assert(fit.success);
	assert(std::abs(fit.focalSubstitution - 0.0001390646L) < 5e-9L);
	assert(std::abs(fit.theta - 0.0005450746L) < 5e-9L);
}

static void testPythonTerminalBenchmarkFits() {
	// Brute-force pooled triplet counts from the same 1 Mb alignment.
	// The expected ages below come from the Python symbolic evaluator and
	// dependency-free Nelder-Mead optimizer with theta fixed at the
	// internal-quartet estimate.
	constexpr long double theta = 0.0005450746L;
	constexpr std::array<long double, 5> countsA = {{
		1995239, 2753, 994, 1014, 0
	}};
	constexpr std::array<long double, 5> countsB = {{
		1995239, 2753, 1014, 994, 0
	}};
	constexpr std::array<long double, 5> countsC = {{
		1993764, 3662, 1278, 1296, 0
	}};
	constexpr std::array<long double, 5> countsD = {{
		1993764, 3662, 1296, 1278, 0
	}};
	auto fitA =
		terminal_branch_length::Estimator<decltype(countsA)>::fit(
			countsA, theta
		);
	auto fitB =
		terminal_branch_length::Estimator<decltype(countsB)>::fit(
			countsB, theta
		);
	auto fitC =
		terminal_branch_length::Estimator<decltype(countsC)>::fit(
			countsC, theta
		);
	auto fitD =
		terminal_branch_length::Estimator<decltype(countsD)>::fit(
			countsD, theta
		);
	assert(fitA.success && fitB.success && fitC.success && fitD.success);
	assert(std::abs(fitA.speciesAges[0] - 0.0002297817790L) < 2e-10L);
	assert(std::abs(fitB.speciesAges[0] - 0.0002297798071L) < 2e-10L);
	assert(std::abs(fitC.speciesAges[0] - 0.0003714579192L) < 2e-10L);
	assert(std::abs(fitD.speciesAges[1] - 0.0009664363384L) < 2e-10L);
}

static void testPositionalPermutationSymmetry() {
	const long double theta = 0.097L;
	const auto unbalanced = branch_length::jc69_msc::unbalanced(
		0.031L, 0.083L, 0.17L, theta);
	const auto balanced = branch_length::jc69_msc::balanced(
		0.031L, 0.047L, 0.17L, theta);
	const auto balancedSwapCherries =
		branch_length::jc69_msc::balanced(
			0.047L, 0.031L, 0.17L, theta);

	for (std::size_t xA = 0; xA < 4; ++xA)
	for (std::size_t xB = 0; xB < 4; ++xB)
	for (std::size_t xC = 0; xC < 4; ++xC)
	for (std::size_t xD = 0; xD < 4; ++xD) {
		const auto original = canonicalClass({{xA, xB, xC, xD}});
		const auto swapAB = canonicalClass({{xB, xA, xC, xD}});
		const auto swapCherries = canonicalClass({{xC, xD, xA, xB}});
		assert(
			std::abs(unbalanced[original] - unbalanced[swapAB]) <
			2e-12L
		);
		assert(
			std::abs(
				balanced[original] -
				balancedSwapCherries[swapCherries]
			) < 2e-12L
		);
	}
}

int main() {
	testPooledCounts();
	testTerminalTraversalCounts();
	testProbabilitiesAndOptimizer();
	testTripletProbabilitiesAndOptimizer();
	testPythonBenchmarkFit();
	testPythonTerminalBenchmarkFits();
	testPositionalPermutationSymmetry();
	std::cout << "branch-length tests passed\n";
}
