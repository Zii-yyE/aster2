#ifndef INTERNAL_BRANCH_LENGTH_HPP
#define INTERNAL_BRANCH_LENGTH_HPP

#include "jc69_msc_probabilities.hpp"
#include "nni_algorithm.hpp"

namespace internal_branch_length {

using std::array;
using std::size_t;
using std::string;
using std::vector;

struct FitResult {
	bool success = false;
	long double logLikelihood =
		-std::numeric_limits<long double>::infinity();
	long double focalBranchSU = 0;
	long double theta = 0;
	array<long double, 3> speciesAgesSU{};
};

template<typename PatternCounts> class Estimator {
	static constexpr array<array<unsigned char, 4>, 15> CLASSES = {{
		{{0, 0, 0, 0}}, {{0, 0, 0, 1}}, {{0, 0, 1, 0}},
		{{0, 0, 1, 1}}, {{0, 0, 1, 2}}, {{0, 1, 0, 0}},
		{{0, 1, 0, 1}}, {{0, 1, 0, 2}}, {{0, 1, 1, 0}},
		{{0, 1, 1, 1}}, {{0, 1, 1, 2}}, {{0, 1, 2, 0}},
		{{0, 1, 2, 1}}, {{0, 1, 2, 2}}, {{0, 1, 2, 3}}
	}};

	static long double jc69DistanceSU(
		long double mismatchFraction
	) noexcept {
		if (mismatchFraction < 0 || mismatchFraction >= 0.75L)
			return -1;
		long double jc69LogArgument = 1 - 4 * mismatchFraction / 3;
		return jc69LogArgument > 0 ?
			-0.75L * std::log(jc69LogArgument) : -1;
	}

	static array<long double, 6> pairwiseDistancesSU(
		PatternCounts const& patternCounts
	) noexcept {
		long double totalCount = 0;
		for (long double count : patternCounts) totalCount += count;

		array<long double, 6> distancesSU{};
		size_t iPair = 0;
		for (size_t i = 0; i < 4; ++i) {
			for (size_t j = i + 1; j < 4; ++j) {
				long double mismatchCount = 0;
				for (size_t k = 0; k < CLASSES.size(); ++k)
					if (CLASSES[k][i] != CLASSES[k][j])
						mismatchCount += patternCounts[k];
				distancesSU[iPair++] =
					jc69DistanceSU(mismatchCount / totalCount);
			}
		}
		return distancesSU;
	}

	static array<long double, 4> initialLogParameters(
		PatternCounts const& patternCounts
	) noexcept {
		array<long double, 6> distancesSU =
			pairwiseDistancesSU(patternCounts);
		long double validDistanceSumSU = 0;
		size_t nValidDistances = 0;
		for (long double value : distancesSU) {
			if (value >= 0) {
				validDistanceSumSU += value;
				++nValidDistances;
			}
		}
		long double fallbackDistanceSU = nValidDistances ?
			std::max(
				validDistanceSumSU / nValidDistances, 0.05L
			) : 0.1L;
		for (long double& value : distancesSU)
			if (value < 0) value = fallbackDistanceSU;

		// Pair order is AB, AC, AD, BC, BD, CD. The first three distances
		// initialize the ordered divergence ages of (((A,B),C),D).
		long double ageABSU =
			std::max(distancesSU[0] / 2, 1e-6L);
		long double ageABCSU =
			std::max(distancesSU[1] / 2, ageABSU + 1e-6L);
		long double ageABCDSU =
			std::max(distancesSU[2] / 2, ageABCSU + 1e-6L);
		return {{
			std::log(ageABSU),
			std::log(ageABCSU - ageABSU),
			std::log(ageABCDSU - ageABCSU),
			std::log(std::max(fallbackDistanceSU, 1e-3L))
		}};
	}

	static bool unpackLogParameters(
		vector<long double> const& logParameters,
		array<long double, 3>& speciesAgesSU,
		long double& theta
	) noexcept {
		// ASTER2 is compiled with -Ofast, whose finite-math assumptions make
		// NaN sentinels unsafe. Reject extreme log coordinates before exp()
		// so every subsequently evaluated model parameter stays finite.
		constexpr long double LOG_MIN = -30;
		constexpr long double LOG_MAX = 10;
		array<long double, 4> positiveParameters{};
		for (size_t i = 0; i < positiveParameters.size(); ++i) {
			if (
				logParameters[i] < LOG_MIN ||
				logParameters[i] > LOG_MAX
			)
				return false;
			positiveParameters[i] = std::exp(logParameters[i]);
		}
		speciesAgesSU[0] = positiveParameters[0];
		speciesAgesSU[1] =
			speciesAgesSU[0] + positiveParameters[1];
		speciesAgesSU[2] =
			speciesAgesSU[1] + positiveParameters[2];
		theta = positiveParameters[3];
		return true;
	}

	static long double negativeLogLikelihood(
		PatternCounts const& patternCounts,
		vector<long double> const& logParameters
	) noexcept {
		array<long double, 3> speciesAgesSU{};
		long double theta = 0;
		if (!unpackLogParameters(
			logParameters, speciesAgesSU, theta
		))
			return std::numeric_limits<long double>::infinity();

		auto patternProbabilities =
			branch_length::jc69_msc::rootedQuartet(
				speciesAgesSU[0], speciesAgesSU[1],
				speciesAgesSU[2], theta
			);
		long double logLikelihood = 0;
		for (size_t i = 0; i < patternCounts.size(); ++i) {
			long double patternProbability = patternProbabilities[i];
			if (!(patternProbability > 0))
				return std::numeric_limits<long double>::infinity();
			if (patternCounts[i] != 0)
				logLikelihood +=
					patternCounts[i] * std::log(patternProbability);
		}
		return -logLikelihood;
	}

	static std::tuple<vector<long double>, long double, size_t>
	nelderMead(
		PatternCounts const& patternCounts,
		vector<long double> const& initialLogParameters,
		size_t maxIterations = 200
	) noexcept {
		constexpr long double STEP = 0.5L;
		constexpr long double X_TOL = 1e-4L;
		constexpr long double F_TOL = 1e-6L;
		size_t n = initialLogParameters.size();
		vector<vector<long double>> simplex(
			n + 1, initialLogParameters
		);
		for (size_t i = 0; i < n; ++i)
			simplex[i + 1][i] += STEP;
		vector<long double> values;
		for (auto const& point : simplex)
			values.push_back(
				negativeLogLikelihood(patternCounts, point)
			);

		size_t iteration = 0;
		while (iteration < maxIterations) {
			vector<size_t> order(n + 1);
			for (size_t i = 0; i <= n; ++i) order[i] = i;
			std::sort(
				order.begin(), order.end(),
				[&](size_t i, size_t j) {
					return values[i] < values[j];
				}
			);
			vector<vector<long double>> sortedSimplex;
			vector<long double> sortedValues;
			for (size_t index : order) {
				sortedSimplex.push_back(simplex[index]);
				sortedValues.push_back(values[index]);
			}
			simplex.swap(sortedSimplex);
			values.swap(sortedValues);

			long double vertexSpan = 0;
			long double valueSpan = 0;
			for (size_t i = 1; i <= n; ++i) {
				valueSpan = std::max(
					valueSpan, std::abs(values[i] - values[0])
				);
				for (size_t j = 0; j < n; ++j) {
					vertexSpan = std::max(
						vertexSpan,
						std::abs(simplex[i][j] - simplex[0][j])
					);
				}
			}
			if (vertexSpan <= X_TOL && valueSpan <= F_TOL) break;

			vector<long double> centroid(n, 0);
			for (size_t i = 0; i < n; ++i)
				for (size_t j = 0; j < n; ++j)
					centroid[j] += simplex[i][j] / n;
			auto combine = [&](
				vector<long double> const& from,
				vector<long double> const& toward,
				long double scale
			) {
				vector<long double> result(n);
				for (size_t j = 0; j < n; ++j)
					result[j] =
						from[j] + scale * (toward[j] - from[j]);
				return result;
			};

			vector<long double> reflected =
				combine(centroid, simplex[n], -1);
			long double reflectedValue =
				negativeLogLikelihood(patternCounts, reflected);
			if (
				values[0] <= reflectedValue &&
				reflectedValue < values[n - 1]
			) {
				simplex[n] = std::move(reflected);
				values[n] = reflectedValue;
				++iteration;
				continue;
			}
			if (reflectedValue < values[0]) {
				vector<long double> expanded =
					combine(centroid, reflected, 2);
				long double expandedValue =
					negativeLogLikelihood(patternCounts, expanded);
				if (expandedValue < reflectedValue) {
					simplex[n] = std::move(expanded);
					values[n] = expandedValue;
				}
				else {
					simplex[n] = std::move(reflected);
					values[n] = reflectedValue;
				}
				++iteration;
				continue;
			}

			vector<long double> contracted =
				combine(centroid, simplex[n], 0.5L);
			long double contractedValue =
				negativeLogLikelihood(patternCounts, contracted);
			if (contractedValue < values[n]) {
				simplex[n] = std::move(contracted);
				values[n] = contractedValue;
				++iteration;
				continue;
			}

			for (size_t i = 1; i <= n; ++i) {
				simplex[i] =
					combine(simplex[0], simplex[i], 0.5L);
				values[i] =
					negativeLogLikelihood(patternCounts, simplex[i]);
			}
			++iteration;
		}

		size_t best = 0;
		for (size_t i = 1; i <= n; ++i)
			if (values[i] < values[best]) best = i;
		return {simplex[best], values[best], iteration};
	}

public:
	static FitResult fit(
		PatternCounts const& patternCounts
	) noexcept {
		long double totalCount = 0;
		for (long double count : patternCounts) totalCount += count;
		if (!(totalCount > 0) || !std::isfinite(totalCount)) return {};

		array<long double, 4> initialLogParameters =
			Estimator::initialLogParameters(patternCounts);
		constexpr array<long double, 5> THETA_SCALES = {{
			0.01L, 0.1L, 1, 10, 100
		}};
		vector<long double> bestLogParameters;
		long double bestNegativeLogLikelihood =
			std::numeric_limits<long double>::infinity();
		for (long double thetaScale : THETA_SCALES) {
			vector<long double> restartLogParameters(
				initialLogParameters.begin(), initialLogParameters.end()
			);
			restartLogParameters.back() += std::log(thetaScale);
			auto [
				fittedLogParameters,
				fittedNegativeLogLikelihood,
				iterations
			] = nelderMead(patternCounts, restartLogParameters);
			(void)iterations;
			if (
				fittedNegativeLogLikelihood <
				bestNegativeLogLikelihood
			) {
				bestNegativeLogLikelihood =
					fittedNegativeLogLikelihood;
				bestLogParameters =
					std::move(fittedLogParameters);
			}
		}
		if (bestLogParameters.empty()) return {};

		array<long double, 3> speciesAgesSU{};
		long double theta = 0;
		if (!unpackLogParameters(
			bestLogParameters, speciesAgesSU, theta
		)) return {};
		long double focalBranchSU =
			speciesAgesSU[1] - speciesAgesSU[0];
		if (focalBranchSU <= 0 || theta <= 0) return {};
		return {
			true, -bestNegativeLogLikelihood, focalBranchSU,
			theta, speciesAgesSU
		};
	}
};

template<typename C> class PooledPatternCounts {
public:
	using Color = C;
	using PatternCounts = typename Color::PatternCounts;
	static PooledPatternCounts const ZERO;
	static inline string const FULL_NAME =
		"Rooted internal branch lengths (MSC+JC69)";

private:
	PatternCounts patternCounts{};
	static inline vector<common::AnnotatedBinaryTree::Node*> fittedNodes;
	static inline vector<long double> fittedLocalThetas;
	static inline size_t attemptedFits = 0;
	static inline size_t failedFits = 0;

	explicit PooledPatternCounts(
		PatternCounts const& countsToPool
	) noexcept : patternCounts(countsToPool) {}

public:
	PooledPatternCounts() noexcept = default;

	static void initialize() noexcept {
		fittedNodes.clear();
		fittedLocalThetas.clear();
		attemptedFits = 0;
		failedFits = 0;
	}

	static array<PooledPatternCounts, 3> map(
		Color& color, size_t iElement
	) noexcept {
		PatternCounts pair23Counts{}, pair13Counts{}, pair12Counts{};
		// Color 0 is the root-side part containing the outgroup. For each
		// unrooted split, order the opposite pair as the youngest cherry:
		// (((A,B),C),D).
		color.elementAccumulateQuartetPatternCounts(
			iElement, {{2, 3, 1, 0}}, pair23Counts
		);
		color.elementAccumulateQuartetPatternCounts(
			iElement, {{1, 3, 2, 0}}, pair13Counts
		);
		color.elementAccumulateQuartetPatternCounts(
			iElement, {{1, 2, 3, 0}}, pair12Counts
		);
		return {{
			PooledPatternCounts(pair23Counts),
			PooledPatternCounts(pair13Counts),
			PooledPatternCounts(pair12Counts)
		}};
	}

	static PooledPatternCounts reduce(
		PooledPatternCounts const& a,
		PooledPatternCounts const& b
	) noexcept {
		PooledPatternCounts result;
		for (size_t i = 0; i < result.patternCounts.size(); ++i)
			result.patternCounts[i] =
				a.patternCounts[i] + b.patternCounts[i];
		return result;
	}

	static void annotate(
		common::AnnotatedBinaryTree::Node* node,
		PooledPatternCounts const& primary,
		PooledPatternCounts const&,
		PooledPatternCounts const&
	) noexcept {
		++attemptedFits;
		FitResult fit = Estimator<PatternCounts>::fit(
			primary.patternCounts
		);
		if (!fit.success) {
			++failedFits;
			common::LogInfo(-1).log()
				<< "Rooted branch-length optimization failed for one "
				"internal edge." << std::endl;
			return;
		}
		node->set("length", (double)fit.focalBranchSU);
		node->set("theta_local", (double)fit.theta);
		node->set(
			"branch_length_log_likelihood",
			(double)fit.logLikelihood
		);
		fittedNodes.push_back(node);
		fittedLocalThetas.push_back(fit.theta);
	}

	static void finalize(common::AnnotatedBinaryTree& tree) {
		if (attemptedFits == 0) return;
		if (fittedLocalThetas.empty()) {
			throw std::runtime_error(
				"Rooted internal branch-length optimization failed "
				"for every edge."
			);
		}
		std::sort(fittedLocalThetas.begin(), fittedLocalThetas.end());
		size_t medianIndex = fittedLocalThetas.size() / 2;
		long double globalTheta = fittedLocalThetas.size() % 2 ?
			fittedLocalThetas[medianIndex] :
			(
				fittedLocalThetas[medianIndex - 1] +
				fittedLocalThetas[medianIndex]
			) / 2;
		tree.set("theta", (double)globalTheta);
		for (auto* node : fittedNodes) {
			long double branchSU = node->get<double>("length");
			node->set(
				"length_cu",
				(double)(2 * branchSU / globalTheta)
			);
		}
		common::LogInfo(0).log()
			<< "Estimated global theta (median of "
			<< fittedLocalThetas.size()
			<< " rooted internal-edge fits): "
			<< (double)globalTheta << std::endl;
		if (failedFits) {
			common::LogInfo(-1).log()
				<< failedFits << " of " << attemptedFits
				<< " rooted internal branch-length fits failed."
				<< std::endl;
		}
	}
};

template<typename Color>
PooledPatternCounts<Color> const PooledPatternCounts<Color>::ZERO =
	PooledPatternCounts<Color>();

template<typename Color> class Procedure {
public:
	using Data = typename Color::SharedConstData;
	using Support = PooledPatternCounts<Color>;
	using TraversalAttributes =
		nni_algorithm::StepwiseColorQuadripartitionScoreDefaultAttributes<
			Color, Support
		>;
	using Traversal =
		nni_algorithm::StepwiseColorQuadripartitionScore<
			TraversalAttributes
		>;

	static void annotate(
		Data const& data,
		common::AnnotatedBinaryTree& tree,
		size_t nThreads,
		int verbose
	) {
		common::LogInfo(verbose).log()
			<< "Annotating " << Support::FULL_NAME << " ..."
			<< std::endl;
		Color color(data, Color::SiteView::ALL_SITES);
		Support::initialize();
		typename Traversal::ThreadPool threadPool(
			nThreads, 0, data.nElements()
		);
		Traversal traversal(color, threadPool, tree, verbose);
		traversal.labelTree();
		Support::finalize(tree);
	}
};

} // namespace internal_branch_length

#endif
