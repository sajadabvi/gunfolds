selfloops = []
no_selfloops = []

network1_GT_selfloop = {1: {1: 1, 2: 1, 5: 1}, 2: {2: 1, 3: 1, 1: 1}, 3: {3: 1, 4: 1}, 4: {4: 1, 5: 1},
                        5: {5: 1}}
network1_GT = {1: {2: 1, 5: 1}, 2: {3: 1, 1: 1}, 3: {4: 1}, 4: {5: 1}, 5: {}}
selfloops.append(network1_GT_selfloop)
no_selfloops.append(network1_GT)

network2_GT_selfloop = {1: {1: 1, 2: 1, 5: 1}, 2: {2: 1, 3: 1, 1: 1}, 3: {2: 1, 3: 1, 4: 1}, 4: {4: 1, 5: 1},
                        5: {5: 1}}
network2_GT = {1: {2: 1, 5: 1}, 2: {3: 1, 1: 1}, 3: {2: 1, 4: 1}, 4: {5: 1}, 5: {}}
selfloops.append(network2_GT_selfloop)
no_selfloops.append(network2_GT)

network3_GT_selfloop = {1: {1: 1, 2: 1, 5: 1}, 2: {2: 1, 3: 1, 1: 1}, 3: {3: 1, 4: 1}, 4: {3: 1, 4: 1, 5: 1},
                        5: {5: 1}}
network3_GT = {1: {2: 1, 5: 1}, 2: {3: 1, 1: 1}, 3: {4: 1}, 4: {3: 1, 5: 1}, 5: {}}
selfloops.append(network3_GT_selfloop)
no_selfloops.append(network3_GT)

network4_GT_selfloop = {1: {4: 1, 8: 1, 6: 1, 1: 1}, 2: {2: 1, 3: 1}, 3: {2: 1, 3: 1},
                        4: {4: 1, 2: 1, 7: 1, 9: 1, 5: 1}, 5: {5: 1, 4: 1, 6: 1},
                        6: {6: 1, 10: 1}, 7: {7: 1, 3: 1, 10: 1}, 8: {8: 1, 2: 1, 9: 1}, 9: {9: 1, 8: 1, 6: 1},
                        10: {10: 1, 6: 1}}
network4_GT = {1: {4: 1, 8: 1, 6: 1}, 2: {3: 1}, 3: {2: 1}, 4: {2: 1, 7: 1, 9: 1, 5: 1}, 5: {4: 1, 6: 1},
               6: {10: 1}, 7: {3: 1, 10: 1}, 8: {2: 1, 9: 1}, 9: {8: 1, 6: 1}, 10: {6: 1}}
selfloops.append(network4_GT_selfloop)
no_selfloops.append(network4_GT)

network5_GT_selfloop = {1: {1: 1, 3: 1}, 2: {2: 1, 4: 1}, 3: {3: 1, 4: 1, 5: 1}, 4: {4: 1, 3: 1}, 5: {5: 1}}
network5_GT = {1: {3: 1}, 2: {4: 1}, 3: {4: 1, 5: 1}, 4: {3: 1}, 5: {}}
selfloops.append(network5_GT_selfloop)
no_selfloops.append(network5_GT)

network6_GT_selfloop = {1: {1: 1, 3: 1}, 2: {2: 1, 3: 1}, 3: {3: 1, 4: 1}, 4: {4: 1, 3: 1, 5: 1},
                        5: {5: 1, 7: 1, 8: 1, 6: 1},
                        6: {6: 1}, 7: {7: 1}, 8: {8: 1}}
network6_GT = {1: {3: 1}, 2: {3: 1}, 3: {4: 1}, 4: {3: 1, 5: 1}, 5: {7: 1, 8: 1, 6: 1}, 6: {}, 7: {}, 8: {}}
selfloops.append(network6_GT_selfloop)
no_selfloops.append(network6_GT)

network7_GT_selfloop = {1: {1: 1, 2: 1}, 2: {2: 1, 3: 1}, 3: {3: 1, 4: 1}, 4: {4: 1, 5: 1},
                        5: {5: 1, 2: 1, 6: 1},
                        6: {6: 1}}
network7_GT = {1: {2: 1}, 2: {3: 1}, 3: {4: 1}, 4: {5: 1}, 5: {2: 1, 6: 1}, 6: {}}
selfloops.append(network7_GT_selfloop)
no_selfloops.append(network7_GT)

network8_GT_selfloop = {1: {1: 1, 2: 1}, 2: {2: 1, 3: 1}, 3: {3: 1, 4: 1, 8: 1}, 4: {4: 1, 5: 1, 6: 1},
                        5: {5: 1, 2: 1}, 6: {6: 1, 7: 1}, 7: {7: 1, 5: 1}, 8: {8: 1}}
network8_GT = {1: {2: 1}, 2: {3: 1}, 3: {4: 1, 8: 1}, 4: {5: 1, 6: 1}, 5: {5: 1, 2: 1}, 6: {7: 1},
               7: {5: 1}, 8: {}}
selfloops.append(network8_GT_selfloop)
no_selfloops.append(network8_GT)

network9_GT_selfloop = {1: {1: 1, 2: 1}, 2: {2: 1, 3: 1}, 3: {3: 1, 4: 1}, 4: {4: 1, 5: 1}, 5: {5: 1, 2: 1},
                        6: {6: 1, 7: 1, 9: 1}, 7: {7: 1, 8: 1}, 8: {8: 1, 4: 1}, 9: {9: 1}}
network9_GT = {1: {2: 1}, 2: {3: 1}, 3: {4: 1}, 4: {5: 1}, 5: {2: 1}, 6: {7: 1, 9: 1}, 7: {8: 1}, 8: {4: 1},
               9: {}}
selfloops.append(network9_GT_selfloop)
no_selfloops.append(network9_GT)


def simp_nets(nn, selfloop=True):
    if selfloop:
        return selfloops[nn - 1]
    else:
        return no_selfloops[nn - 1]