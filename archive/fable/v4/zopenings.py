"""Likely AI Chessathon curated starting positions.

These are position-only data (no third-party engine code or evaluations).  They are drawn from
publicly available balanced-opening corpora whose named positions overlap the openings displayed
on public AI Chessathon game/team pages.  Fable uses them only to warm its own TT during the free
initialisation window; every returned move still comes from Fable's own search and is legality
checked against the live FEN.
"""

# Keep full six-field FENs because the halfmove clock can matter to search.  Duplicates are cheap
# but are avoided here to make the init budget deterministic.
SEED_POSITIONS: tuple[tuple[str, str], ...] = (
    ("Sicilian Sveshnikov", "r1bqkb1r/5ppp/p1np1n2/1p2p3/4P3/N1N5/PPP1BPPP/R1BQK2R w KQkq - 0 9"),
    ("Sicilian Taimanov", "r1b1kb1r/pp3ppp/1qnppn2/8/3NP3/2N1B3/PPP1BPPP/R2QK2R w KQkq - 2 8"),
    ("Sicilian Classical", "r1bqkb1r/pp3ppp/2nppn2/8/3NP3/2N5/PPP1BPPP/R1BQK2R w KQkq - 0 7"),
    ("Sicilian Dragon", "r1bqkb1r/pp3ppp/2nppn2/8/3NP3/2N1B3/PPP2PPP/R2QKB1R w KQkq - 0 7"),
    ("Sicilian Najdorf", "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6"),
    ("Sicilian Closed", "r1bqk1nr/pp1pppbp/2n3p1/2p5/4P3/2NP2P1/PPP2PBP/R1BQK1NR b KQkq - 0 5"),
    ("Italian Game", "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
    ("Two Knights Defence", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
    ("Ruy Lopez, Closed", "r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 1 5"),
    ("Ruy Lopez, main", "r1bqk2r/1pppbppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 3 6"),
    ("King's Indian Classical", "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R b KQ - 3 6"),
    ("King's Indian", "rnbqk2r/ppp1ppbp/3p1np1/8/2PPP3/2N5/PP3PPP/R1BQKBNR w KQkq - 0 5"),
    ("French Advance", "rnbqkbnr/ppp2ppp/4p3/3pP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3"),
    ("French Classical", "rnbqk2r/ppp1bppp/4pn2/3p4/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 4 5"),
    ("French Winawer", "rnbqk2r/ppp2ppp/4pn2/3p4/1b1PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 3 5"),
    ("French Tarrasch", "rnbqkb1r/pp1n1ppp/4p3/2ppP3/3P4/2PB4/PP1N1PPP/R1BQK1NR b KQkq - 0 6"),
    ("Caro-Kann Advance", "rnbqkbnr/pp2pppp/2p5/3pP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3"),
    ("Caro-Kann Classical", "rnbqkb1r/pp2pppp/2p2n2/3p4/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 2 4"),
    ("Caro-Kann Exchange", "r1bqkb1r/pp2pppp/2n2n2/3p4/3P4/2PB4/PP3PPP/RNBQK1NR w KQkq - 1 6"),
    ("English Symmetrical", "rnbqkbnr/pp1ppppp/8/2p5/2P5/5N2/PP1PPPPP/RNBQKB1R b KQkq - 1 2"),
    ("Slav Defence", "rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 2 4"),
    ("Semi-Slav Defence", "rnbqkb1r/pp3ppp/2p2n2/3pp3/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq e6 0 5"),
    ("Nimzo-Indian Defence", "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4"),
    ("Scotch Game", "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq d3 0 3"),
    ("Petroff Defence", "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"),
    ("Pirc Defence", "rnbqkb1r/ppp1pppp/3p1n2/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 2 3"),
    ("London System", "rnbqkb1r/ppp1pppp/3p1n2/8/3P1B2/5N2/PPP1PPPP/RN1QKB1R b KQkq - 3 3"),
    ("Catalan Opening", "rnbqkb1r/pppp1ppp/4pn2/8/2PP4/6P1/PP2PP1P/RNBQKBNR b KQkq - 0 3"),
    ("Grunfeld Defence", "rnbqkb1r/ppp1pp1p/5np1/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq d6 0 4"),
    ("Reti Opening", "rnbqkbnr/ppp1pppp/8/3p4/8/5NP1/PPPPPP1P/RNBQKB1R b KQkq - 0 2"),
    ("Queen's Indian Defence", "rnbqkb1r/p1pp1ppp/1p2pn2/8/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 5"),
    ("Dutch Stonewall", "rnbqkb1r/pppp2pp/4pn2/5p2/2PP4/6P1/PP2PP1P/RNBQKBNR w KQkq - 0 4"),
    ("Four Knights Game", "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 4 4"),
)
