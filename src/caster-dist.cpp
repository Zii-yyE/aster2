#include "driver.hpp"
#include "common.hpp"
#include "caster.hpp"

using namespace std::string_literals;
using namespace std;

using Caster = caster::Color<caster::StepwiseColorDefaultAttributes<unsigned short, unsigned short> >;

void run(Caster::SharedConstData &data) {
	Caster caster(data);
	size_t n = common::taxonName2ID.nTaxa();
	size_t k = data.nElements();
	vector<vector<vector<double> > > scores(n, vector<vector<double> >(n, vector<double>(k, 0.0)));
	
	for (int i = 0; i < n; ++i) {
		for (int x = 0; x < k; ++x) {
			caster.elementSetTaxonColor(x, i, 0);
			caster.elementSetTaxonColor(x, i, 3);
		}
	}

	for (int i = 0; i < n; ++i) {
		for (int x = 0; x < k; ++x) {
			caster.elementClearTaxonColor(x, i, 0);
			caster.elementClearTaxonColor(x, i, 3);
			caster.elementSetTaxonColor(x, i, 1);
		}

		for (int j = 0; j < i; ++j) {
			for (int x = 0; x < k; ++x) {
				caster.elementClearTaxonColor(x, j, 0);
				caster.elementClearTaxonColor(x, j, 3);
				caster.elementSetTaxonColor(x, j, 2);
				scores[i][j][x] = scores[j][i][x] = caster.elementQuadripartitionScores(x)[0];
				array<double, 3> s = caster.elementQuadripartitionScores(x);
				caster.elementClearTaxonColor(x, j, 2);
				caster.elementSetTaxonColor(x, j, 0);
				caster.elementSetTaxonColor(x, j, 3);
			}
		}

		for (int x = 0; x < k; ++x) {
			caster.elementSetTaxonColor(x, i, 0);
			caster.elementSetTaxonColor(x, i, 3);
			caster.elementClearTaxonColor(x, i, 1);
		}
	}

	for (int x = 0; x < k; ++x) {
		if (x > 0) cout << endl;
		cout << n << endl;
		for (int i = 0; i < n; ++i) {
			cout << common::taxonName2ID[i];
			for (int j = 0; j < n; ++j) {
				cout << "\t" << scores[i][j][x];
			}
			cout << endl;
		}
	}
}

int main(int argc, char* argv[]) {
	using std::string;
	using std::size_t;
	using Clock = std::chrono::high_resolution_clock;
	using Driver = caster::Driver<true>;

	std::pair<string, string> programNames = Driver::programNames();
	ARG.set("SHORT_NAME", programNames.first);
	ARG.set("FULL_NAME", programNames.second);
	
	ARG.addArgument('h', "help", "flag", "Display help message", 6, true);
	ARG.addArgument('i', "input", "string", "Input file path", 5);
	//ARG.addArgument('o', "output", "string", "Output file path, print to stdout if not provided", 5, true);
	ARG.addArgument('a', "mapping", "string", "Mapping file path, a list of gene/speicesman name to taxon name maps, each line contains one gene/speicesman name followed by one taxon name separated by a space or tab", 3, true, false);
	ARG.addArgument('\0', "verbose", "integer", "Verbose level", 0, true, true, "4");
	ARG.addArgument('\0', "no-log", "flag", "Don't generate log file", 1, true);
	ARG.addArgument('\0', "log", "string", "Log file path", 0, true, true, "log.txt");
	ARG.addArgument('\0', "log-verbose", "integer", "Verbose level in log file", 0, true, true, "5");

	ARG.addArgument('\0', "chunk", "integer", "The maximum number of sites in each local aligment block for parameter estimation", 0, true, true, "500");

	ARG.parse(argc, argv);
	common::LogInfo::setVerbose(ARG.has("no-log") ? nullptr : new std::ofstream(ARG.get<string>("log")), ARG.get<size_t>("log-verbose"), ARG.get<size_t>("verbose"));
	ARG.print();

	if (ARG.has("root")) common::taxonName2ID[ARG.get<string>("root")];
	ARG.log() << "Parsing input file(s)..." << endl;
	Caster::SharedConstData data = caster::DriverHelper::read<Caster::SharedConstData>();
	ARG.log() << "Start running ..." << endl;
	run(data);

	return 0;
}