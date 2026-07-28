#ifndef BRANCH_LENGTH_HPP
#define BRANCH_LENGTH_HPP

#include "jc69_msc_probabilities.hpp"
#include "nni_algorithm.hpp"

namespace branch_length {

using std::array;
using std::size_t;
using std::string;
using std::vector;

struct FitResult {
	bool success = false;
	long double logLikelihood = -std::numeric_limits<long double>::infinity();
	long double focalSubstitution = 0;
	long double theta = 0;
};

template<typename PatternCounts> class Estimator {
	static constexpr array<array<unsigned char, 4>, 15> CLASSES = {{
		{{0, 0, 0, 0}}, {{0, 0, 0, 1}}, {{0, 0, 1, 0}}, {{0, 0, 1, 1}},
		{{0, 0, 1, 2}}, {{0, 1, 0, 0}}, {{0, 1, 0, 1}}, {{0, 1, 0, 2}},
		{{0, 1, 1, 0}}, {{0, 1, 1, 1}}, {{0, 1, 1, 2}}, {{0, 1, 2, 0}},
		{{0, 1, 2, 1}}, {{0, 1, 2, 2}}, {{0, 1, 2, 3}}
	}};

	static long double correctedDistance(long double mismatchFraction) noexcept {
		if (mismatchFraction < 0 || mismatchFraction >= 0.75L)
			return std::numeric_limits<long double>::quiet_NaN();
		long double remaining = 1 - 4 * mismatchFraction / 3;
		return remaining > 0 ? -0.75L * std::log(remaining) :
			std::numeric_limits<long double>::quiet_NaN();
	}

	static array<long double, 6> initialParameters(PatternCounts const& counts) noexcept {
		long double total = 0;
		for (long double count : counts) total += count;
		array<long double, 6> distances{};
		size_t iPair = 0;
		for (size_t i = 0; i < 4; i++) for (size_t j = i + 1; j < 4; j++) {
			long double mismatch = 0;
			for (size_t k = 0; k < CLASSES.size(); k++)
				if (CLASSES[k][i] != CLASSES[k][j]) mismatch += counts[k];
			distances[iPair++] = correctedDistance(mismatch / total);
		}

		long double validSum = 0;
		size_t nValid = 0;
		for (long double value : distances) if (std::isfinite(value)) {
			validSum += value;
			nValid++;
		}
		long double fallback = nValid ? std::max(validSum / nValid, 0.05L) : 0.1L;
		for (long double& value : distances) if (!std::isfinite(value)) value = fallback;

		// Pair order is AB, AC, AD, BC, BD, CD.
		long double dAB = distances[0], dAC = distances[1], dAD = distances[2];
		long double dBC = distances[3], dBD = distances[4], dCD = distances[5];
		long double a = std::max((dAB + dAC - dBC) / 2, 1e-6L);
		long double b = std::max((dAB + dBC - dAC) / 2, 1e-6L);
		long double c = std::max((dCD + dAC - dAD) / 2, 1e-6L);
		long double d = std::max((dCD + dAD - dAC) / 2, 1e-6L);
		long double t = std::max((
			(dAC - a - c) + (dAD - a - d) +
			(dBC - b - c) + (dBD - b - d)) / 4, 1e-6L);
		return {{std::log(a), std::log(b), std::log(c), std::log(d),
			std::log(t), std::log(std::max(fallback, 1e-3L))}};
	}

	static long double objective(PatternCounts const& counts, vector<long double> const& params) noexcept {
		array<long double, 6> positive{};
		for (size_t i = 0; i < positive.size(); i++) {
			positive[i] = std::exp(params[i]);
			if (!std::isfinite(positive[i]) || positive[i] <= 0)
				return std::numeric_limits<long double>::infinity();
		}
		auto probabilities = jc69_msc::probabilities(
			positive[0], positive[1], positive[2], positive[3], positive[4], positive[5]);
		long double logLikelihood = 0;
		for (size_t i = 0; i < counts.size(); i++) {
			long double probability = probabilities[i];
			if (!std::isfinite(probability) || probability <= 0)
				return std::numeric_limits<long double>::infinity();
			if (counts[i] != 0) logLikelihood += counts[i] * std::log(probability);
		}
		return -logLikelihood;
	}

	static std::tuple<vector<long double>, long double, size_t> nelderMead(
		PatternCounts const& counts,
		vector<long double> const& initial,
		size_t maxIterations = 200
	) noexcept {
		constexpr long double STEP = 0.5L, X_TOL = 1e-4L, F_TOL = 1e-6L;
		size_t n = initial.size();
		vector<vector<long double> > simplex(n + 1, initial);
		for (size_t i = 0; i < n; i++) simplex[i + 1][i] += STEP;
		vector<long double> values;
		for (auto const& point : simplex) values.push_back(objective(counts, point));

		size_t iteration = 0;
		while (iteration < maxIterations) {
			vector<size_t> order(n + 1);
			for (size_t i = 0; i <= n; i++) order[i] = i;
			std::sort(order.begin(), order.end(), [&](size_t i, size_t j) { return values[i] < values[j]; });
			vector<vector<long double> > sortedSimplex;
			vector<long double> sortedValues;
			for (size_t index : order) {
				sortedSimplex.push_back(simplex[index]);
				sortedValues.push_back(values[index]);
			}
			simplex.swap(sortedSimplex);
			values.swap(sortedValues);

			long double vertexSpan = 0, valueSpan = 0;
			for (size_t i = 1; i <= n; i++) {
				valueSpan = std::max(valueSpan, std::abs(values[i] - values[0]));
				for (size_t j = 0; j < n; j++)
					vertexSpan = std::max(vertexSpan, std::abs(simplex[i][j] - simplex[0][j]));
			}
			if (vertexSpan <= X_TOL && valueSpan <= F_TOL) break;

			vector<long double> centroid(n, 0);
			for (size_t i = 0; i < n; i++) for (size_t j = 0; j < n; j++)
				centroid[j] += simplex[i][j] / n;
			auto combine = [&](vector<long double> const& from, vector<long double> const& toward, long double scale) {
				vector<long double> result(n);
				for (size_t j = 0; j < n; j++) result[j] = from[j] + scale * (toward[j] - from[j]);
				return result;
			};

			vector<long double> reflected = combine(centroid, simplex[n], -1);
			long double reflectedValue = objective(counts, reflected);
			if (values[0] <= reflectedValue && reflectedValue < values[n - 1]) {
				simplex[n] = std::move(reflected);
				values[n] = reflectedValue;
				iteration++;
				continue;
			}
			if (reflectedValue < values[0]) {
				vector<long double> expanded = combine(centroid, reflected, 2);
				long double expandedValue = objective(counts, expanded);
				if (expandedValue < reflectedValue) {
					simplex[n] = std::move(expanded);
					values[n] = expandedValue;
				}
				else {
					simplex[n] = std::move(reflected);
					values[n] = reflectedValue;
				}
				iteration++;
				continue;
			}

			vector<long double> contracted = combine(centroid, simplex[n], 0.5L);
			long double contractedValue = objective(counts, contracted);
			if (contractedValue < values[n]) {
				simplex[n] = std::move(contracted);
				values[n] = contractedValue;
				iteration++;
				continue;
			}

			for (size_t i = 1; i <= n; i++) {
				simplex[i] = combine(simplex[0], simplex[i], 0.5L);
				values[i] = objective(counts, simplex[i]);
			}
			iteration++;
		}
		size_t best = 0;
		for (size_t i = 1; i <= n; i++) if (values[i] < values[best]) best = i;
		return {simplex[best], values[best], iteration};
	}

public:
	static FitResult fit(PatternCounts const& counts) noexcept {
		long double total = 0;
		for (long double count : counts) total += count;
		if (!(total > 0) || !std::isfinite(total)) return {};

		array<long double, 6> base = initialParameters(counts);
		constexpr array<long double, 5> THETA_SCALES = {{0.01L, 0.1L, 1, 10, 100}};
		vector<long double> bestPoint;
		long double bestValue = std::numeric_limits<long double>::infinity();
		for (long double thetaScale : THETA_SCALES) {
			vector<long double> start(base.begin(), base.end());
			start.back() += std::log(thetaScale);
			auto [point, value, iterations] = nelderMead(counts, start);
			(void)iterations;
			if (value < bestValue) {
				bestValue = value;
				bestPoint = std::move(point);
			}
		}
		if (bestPoint.empty() || !std::isfinite(bestValue)) return {};
		long double focal = std::exp(bestPoint[4]);
		long double theta = std::exp(bestPoint[5]);
		if (!std::isfinite(focal) || !std::isfinite(theta) || focal <= 0 || theta <= 0) return {};
		return {true, -bestValue, focal, theta};
	}
};

template<typename C> class PooledPatternCounts {
public:
	using Color = C;
	using PatternCounts = typename Color::PatternCounts;
	static PooledPatternCounts const ZERO;
	static inline string const FULL_NAME = "Internal branch lengths (MSC+JC69)";

private:
	PatternCounts counts{};
	static inline vector<common::AnnotatedBinaryTree::Node*> fittedNodes;
	static inline vector<long double> fittedThetas;
	static inline size_t attempted = 0;
	static inline size_t failed = 0;

	explicit PooledPatternCounts(PatternCounts const& counts) noexcept : counts(counts) {}

public:
	PooledPatternCounts() noexcept = default;

	static void initialize() noexcept {
		fittedNodes.clear();
		fittedThetas.clear();
		attempted = 0;
		failed = 0;
	}

	static array<PooledPatternCounts, 3> map(Color& color, size_t iElement) noexcept {
		PatternCounts primary{}, alternative1{}, alternative2{};
		color.elementAccumulateQuartetPatternCounts(iElement, {{0, 1, 2, 3}}, primary);
		color.elementAccumulateQuartetPatternCounts(iElement, {{0, 2, 1, 3}}, alternative1);
		color.elementAccumulateQuartetPatternCounts(iElement, {{0, 3, 1, 2}}, alternative2);
		return {{PooledPatternCounts(primary), PooledPatternCounts(alternative1), PooledPatternCounts(alternative2)}};
	}

	static PooledPatternCounts reduce(PooledPatternCounts const& a, PooledPatternCounts const& b) noexcept {
		PooledPatternCounts result;
		for (size_t i = 0; i < result.counts.size(); i++) result.counts[i] = a.counts[i] + b.counts[i];
		return result;
	}

	static void annotate(
		common::AnnotatedBinaryTree::Node* node,
		PooledPatternCounts const& primary,
		PooledPatternCounts const&,
		PooledPatternCounts const&
	) noexcept {
		attempted++;
		FitResult fit = Estimator<PatternCounts>::fit(primary.counts);
		if (!fit.success) {
			failed++;
			common::LogInfo(-1).log() << "Branch-length optimization failed for one internal edge." << std::endl;
			return;
		}
		node->set("length", (double)fit.focalSubstitution);
		node->set("theta_local", (double)fit.theta);
		node->set("branch_length_log_likelihood", (double)fit.logLikelihood);
		fittedNodes.push_back(node);
		fittedThetas.push_back(fit.theta);
	}

	static void finalize(common::AnnotatedBinaryTree& tree) {
		if (attempted == 0) return;
		if (fittedThetas.empty())
			throw std::runtime_error("Internal branch-length optimization failed for every edge.");
		std::sort(fittedThetas.begin(), fittedThetas.end());
		size_t middle = fittedThetas.size() / 2;
		long double globalTheta = fittedThetas.size() % 2 ?
			fittedThetas[middle] : (fittedThetas[middle - 1] + fittedThetas[middle]) / 2;
		tree.set("theta", (double)globalTheta);
		for (auto* node : fittedNodes) {
			long double substitution = node->get<double>("length");
			node->set("length_cu", (double)(2 * substitution / globalTheta));
		}
		common::LogInfo(0).log() << "Estimated global theta (median of "
			<< fittedThetas.size() << " internal-edge fits): " << (double)globalTheta << std::endl;
		if (failed) common::LogInfo(-1).log() << failed << " of " << attempted
			<< " internal branch-length fits failed." << std::endl;
	}
};

template<typename Color>
PooledPatternCounts<Color> const PooledPatternCounts<Color>::ZERO = PooledPatternCounts<Color>();

template<typename Color> class Procedure {
public:
	using Data = typename Color::SharedConstData;
	using Support = PooledPatternCounts<Color>;
	using TraversalAttributes = nni_algorithm::StepwiseColorQuadripartitionScoreDefaultAttributes<Color, Support>;
	using Traversal = nni_algorithm::StepwiseColorQuadripartitionScore<TraversalAttributes>;

	static void annotate(Data const& data, common::AnnotatedBinaryTree& tree, size_t nThreads, int verbose) {
		common::LogInfo(verbose).log() << "Annotating " << Support::FULL_NAME << " ..." << std::endl;
		Color color(data, Color::SiteView::ALL_SITES);
		Support::initialize();
		typename Traversal::ThreadPool threadPool(nThreads, 0, data.nElements());
		Traversal traversal(color, threadPool, tree, verbose);
		traversal.labelTree();
		Support::finalize(tree);
	}
};

} // namespace branch_length

#endif
