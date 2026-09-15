"""Original root policy over permitted Syzygy DATA, using preinstalled python-chess.

No engine code or external engine evaluations are shipped. DTZ preserves
progress; WDL protects the game result. Ambiguous fifty-move boundaries return
None so the normal search makes the decision.
"""
from pathlib import Path
import hashlib
import json
import time
import chess
import chess.syzygy


def position_key(board):
    return (board.board_fen(), board.turn, board.castling_rights,
            board.ep_square if board.has_legal_en_passant() else None)


class Endgames:
    def __init__(self, directory):
        directory = Path(directory)
        for item in json.loads((directory/'manifest.json').read_text()):
            data = (directory/item['file']).read_bytes()
            if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError('Tablebase integrity check failed: '+item['file'])
        self.tables = chess.syzygy.open_tablebase(str(directory), max_fds=96)
        # Parse/map the complete tiny set during the free init allowance.
        for table in self.tables.wdl.values():
            table.init_table_wdl()
        for table in self.tables.dtz.values():
            table.init_table_dtz()
        self.hits = 0

    def choose(self, board, seen, deadline):
        if len(board.piece_map()) > 4 or board.castling_rights:
            return None
        ranked = []
        ambiguous = False
        try:
            for move in list(board.legal_moves):
                if time.perf_counter() >= deadline:
                    return None
                zeroing = board.is_zeroing(move)
                before_half = board.halfmove_clock
                board.push(move)
                try:
                    if board.is_checkmate():
                        ranked.append(((3, 0), move))
                        continue
                    if (board.is_stalemate() or board.is_insufficient_material()
                        or board.halfmove_clock >= 100 or seen.get(position_key(board), 0) >= 2):
                        ranked.append(((0, 0), move))
                        continue
                    wdl = -self.tables.probe_wdl(board)
                    dtz = self.tables.probe_dtz(board)
                    distance = 1 if zeroing else abs(dtz) + 1
                    outcome = 2 if wdl == 2 else (-2 if wdl == -2 else 0)
                    if not zeroing and abs(wdl) == 2 and before_half + distance >= 99:
                        # DTZ may be rounded by one ply. Search the actual clock
                        # boundary instead of asserting an unsafe table result.
                        ambiguous = True
                        continue
                    progress = -distance if outcome > 0 else (distance if outcome < 0 else 0)
                    ranked.append(((outcome, progress), move))
                finally:
                    board.pop()
            if not ranked:
                return None
            best = max(ranked, key=lambda item: item[0])
            if ambiguous and best[0][0] < 2:
                return None
            self.hits += 1
            return best[1]
        except (KeyError, OSError, ValueError):
            return None
