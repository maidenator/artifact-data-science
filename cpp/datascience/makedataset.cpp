#include <array>
#include <cstdlib>
#include <fstream>
#include "generator.hpp"

int main(int argc, char** argv) {
    const int N = argc > 1 ? std::atoi(argv[1]) : 200000;
    rng::Xoshiro256 rng(20261006);
    std::ofstream out("artifacts.csv");
    out << "artifact_id,level,slot,main_stat,sub_count,cv,has_crit_rate,has_crit_dmg,final_cv\n";

    struct Snap { int level, subCount; float cv; int cr, cd; };

    for (int id = 0; id < N; ++id) {
        Artifact art = generator::generateArtifact(rng);
        std::array<Snap, 6> snaps;
        int n = 0;

        auto take = [&] {
            int cr = 0, cd = 0;
            for (int i = 0; i < art.substatCount; ++i) {
                cr |= art.subStats[i].type == ArtifactSubstat::critRate;
                cd |= art.subStats[i].type == ArtifactSubstat::critDmg;
            }
            snaps[n++] = {art.level, art.substatCount,
                          generator::calculateCritValue(art), cr, cd};
        };

        take();                                   // +0
        while (art.level < 20) {
            generator::upgradeArtifactOnce(art, rng);
            take();                               // +4 ... +20
        }

        float finalCv = snaps[n - 1].cv;
        for (int i = 0; i < n; ++i) {
            out << id << ',' << snaps[i].level << ','
                << static_cast<int>(art.slot) << ','
                << static_cast<int>(art.mainStat.type) << ','
                << snaps[i].subCount << ',' << snaps[i].cv << ','
                << snaps[i].cr << ',' << snaps[i].cd << ','
                << finalCv << '\n';
        }
    }
}