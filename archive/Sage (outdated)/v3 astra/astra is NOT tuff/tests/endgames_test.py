"""Verify shipped tablebase integrity, root policy, and complete conversions."""
import sys,json,time,random
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import chess
from endgames import Endgames,position_key
root=Path(__file__).resolve().parents[1]
tb=Endgames(root/'tablebases');rng=random.Random(7231)
count=0
for i in range(500):
 b=chess.Board(None);sq=rng.sample(range(64),4)
 b.set_piece_at(sq[0],chess.Piece(chess.KING,True));b.set_piece_at(sq[1],chess.Piece(chess.KING,False))
 for j in (2,3):b.set_piece_at(sq[j],chess.Piece(rng.randrange(1,6),bool(rng.randrange(2))))
 b.turn=bool(rng.randrange(2))
 if not b.is_valid() or b.is_game_over():continue
 expected=tb.tables.probe_wdl(b)
 before=b.fen();move=tb.choose(b,Counter({position_key(b):1}),time.perf_counter()+1)
 assert b.fen()==before
 assert move in b.legal_moves,(b.fen(),move)
 b.push(move)
 actual=2 if b.is_checkmate() else -tb.tables.probe_wdl(b)
 assert (actual==expected or (abs(actual)<=1 and abs(expected)<=1)),(before,expected,actual,move)
 count+=1
# Use both sides' DTZ policy in winning elementary endings: no shuffling draws.
conversions=[]
for fen in ['8/8/2k5/8/8/3K4/8/Q7 w - - 0 1',
            '8/8/2k5/8/8/3K4/8/R7 w - - 0 1',
            '8/8/2k5/8/8/3K4/8/BN6 w - - 0 1',
            '8/8/2k5/8/8/3K4/8/BB6 w - - 0 1']:
 b=chess.Board(fen);seen=Counter();moves=0
 assert tb.tables.probe_wdl(b)==2
 while not b.is_game_over(claim_draw=True) and moves<150:
  seen[position_key(b)]+=1
  m=tb.choose(b,seen,time.perf_counter()+1);assert m in b.legal_moves,(b.fen(),m)
  b.push(m);moves+=1
 assert b.is_checkmate(),(fen,b.fen(),moves)
 conversions.append({'fen':fen,'plies_to_mate':moves})
# Expired budget leaves board unchanged and falls back to search.
b=chess.Board(conversions[0]['fen']);fen=b.fen();assert tb.choose(b,{},0) is None;assert b.fen()==fen
report={'table_files':70,'random_wdl_preservation_positions':count,'complete_mating_conversions':conversions,'timeout_restoration':'passed'}
print(json.dumps(report,indent=2),flush=True)
if '--output' in sys.argv:Path(sys.argv[sys.argv.index('--output')+1]).write_text(json.dumps(report,indent=2)+'\n')
