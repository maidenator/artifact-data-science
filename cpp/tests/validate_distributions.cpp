#include <iostream>
#include <random>
#include "../artifact/generator.hpp"

int main() {
    std::random_device rd;
    rng::Xoshiro256 rng(rd());
    const int N = 1'000'000;

    std::array<long, 5> slotCount{};
    std::array<long, 19> gobletMain{};
    std::array<long, 10> subCount{};
    long fourLiners = 0, gobletTotal = 0, totalSubs = 0;

    for (int i = 0; i < N; ++i) {
        Artifact a = generator::generateArtifact(rng);
        slotCount[static_cast<int>(a.slot)]++;
        fourLiners += (a.substatCount == 4);
        if (a.slot == ArtifactSlot::goblet) {
            gobletMain[static_cast<int>(a.mainStat.type)]++;
            gobletTotal++;
        }
        for (int s = 0; s < a.substatCount; ++s) {
            subCount[static_cast<int>(a.subStats[s].type)]++;
            totalSubs++;
        }
    }

    std::cout << "Four-liner rate (expect ~20%): " << 100.0 * fourLiners / N << "%\n";
    std::cout << "Slot shares (expect ~20% each):\n";
    for (int i = 0; i < 5; ++i) std::cout << "  " << i << ": " << 100.0 * slotCount[i] / N << "%\n";
    std::cout << "Goblet main stats (expect HP% 19.25, ATK% 19.25, DEF% 19, elements 5, EM 2.5):\n";
    for (int i = 0; i < 19; ++i)
        if (gobletMain[i]) std::cout << "  " << i << ": " << 100.0 * gobletMain[i] / gobletTotal << "%\n";
    std::cout << "Substat shares:\n";
    for (int i = 0; i < 10; ++i) std::cout << "  " << i << ": " << 100.0 * subCount[i] / totalSubs << "%\n";
}