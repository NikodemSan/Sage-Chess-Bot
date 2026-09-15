"""Deterministic differential checks; run python tests/verify.py from ZIP root."""
import os
os.environ.setdefault('SAGE_TT_BITS', '20')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time, json, random
import numpy as np
import chess
import zboard as zb
import zengine
import zeval as ze
import zsearch as zs

ROOT = Path(__file__).resolve().parents[1]
START = time.perf_counter()
rng = random.Random(927164)
e = zengine.Engine(str(ROOT/'models/sage150m.npz'))
e.search(chess.STARTING_FEN, 10, 20, max_depth=2)
report = {}

# Known exhaustive legal-tree counts, including castling, EP, checks, promotions.
perfts = [
(chess.STARTING_FEN, 4, 197281),
('r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1', 3, 97862),
('8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1', 4, 43238),
('r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1', 3, 9467),
('rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8', 3, 62379),
]
for fen, depth, expected in perfts:
 e.new_game(); e.set_position(fen)
 result=zb.perft(e.bd,e.st,e.undo,e.hs,depth,e.mvbuf)
 assert result==expected,(fen,result,expected)
report['perft_positions']=len(perfts)
report['perft_leaf_nodes']=sum(x[2] for x in perfts)
print('perft passed',flush=True)

# Differential move generation, make/unmake, hashes and deferred NN accumulators
# against independently rebuilt arrays, while retaining multi-ply lazy chains.
boards=[chess.Board(),*[chess.Board(f) for f,_,_ in perfts[1:]],
 chess.Board('4k3/P7/8/8/8/8/7p/4K3 w - - 0 1'),
 chess.Board('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1')]
positions=0; transitions=0
for game in range(24):
 b=boards[game%len(boards)].copy();e.new_game();e.set_position(b.fen(en_passant='fen'))
 made=[]
 for ply in range(70):
  p=int(e.st[zb.ST_PLY]); n=zb.gen_moves(e.bd,e.st,e.mvbuf[p],False)
  before_bd=e.bd.copy();before_st=e.st.copy();moves={}
  for m in list(map(int,e.mvbuf[p,:n])):
   if zb.make_move(e.bd,e.st,e.undo,e.hs,m):
    moves[zb.move_to_uci(m)]=m
    assert e.hs[e.st[zb.ST_HISTLEN]+e.st[zb.ST_PLY]]==zb.compute_hash(e.bd,e.st)
    zb.unmake_move(e.bd,e.st,e.undo,m)
   assert np.array_equal(before_bd,e.bd) and np.array_equal(before_st,e.st)
  assert set(moves)==set(m.uci() for m in b.legal_moves),(b.fen(),set(moves)^set(m.uci() for m in b.legal_moves))
  positions+=1
  if not moves: break
  # Randomly skip evaluations: forces deferred multi-ply neural updates.
  if rng.random()<0.55:
   score=ze.evaluate(e.bd,e.st,e.mode,e.tbl_mg,e.tbl_eg,e.acc,p,e.w1,e.w2,e.b2)
   fresh=np.zeros_like(e.acc);ze.nn_refresh(e.bd,fresh,p,e.w1,e.b1)
   expected=ze.evaluate(e.bd,e.st,e.mode,e.tbl_mg,e.tbl_eg,fresh,p,e.w1,e.w2,e.b2)
   assert score==expected,(b.fen(),score,expected)
   assert np.array_equal(e.acc[p,:,:e.w1.shape[1]],fresh[p,:,:e.w1.shape[1]])
   # A speculative null copies fully evaluated accumulators, and its history
   # horizon must exclude all real positions before the null.
   st0=e.st.copy();h=e.acc.shape[2]
   zb.make_null(e.bd,e.st,e.undo,e.hs);e.acc[p+1]=e.acc[p]
   assert e.st[zb.ST_HALF]==0
   assert e.hs[e.st[zb.ST_HISTLEN]+e.st[zb.ST_PLY]]==zb.compute_hash(e.bd,e.st)
   fresh=np.zeros_like(e.acc);ze.nn_refresh(e.bd,fresh,p+1,e.w1,e.b1)
   assert ze.evaluate(e.bd,e.st,e.mode,e.tbl_mg,e.tbl_eg,e.acc,p+1,e.w1,e.w2,e.b2)==ze.evaluate(e.bd,e.st,e.mode,e.tbl_mg,e.tbl_eg,fresh,p+1,e.w1,e.w2,e.b2)
   zb.unmake_null(e.st,e.undo);assert np.array_equal(e.st,st0)
  uci=rng.choice(list(moves));m=moves[uci]
  ze.nn_push(e.bd,e.st,m,e.acc,p,e.w1)
  assert zb.make_move(e.bd,e.st,e.undo,e.hs,m)
  b.push_uci(uci);made.append(m);transitions+=1
 for m in reversed(made): zb.unmake_move(e.bd,e.st,e.undo,m)
report['random_positions']=positions;report['random_transitions']=transitions
print('random differential checks passed',positions,flush=True)

# Independent implementation of the documented training feature contract.
from sl_spec import features, QA, QB, CP_SCALE
for i in range(100):
 b=chess.Board()
 for _ in range(rng.randrange(80)):
  if b.is_game_over(): break
  b.push(rng.choice(list(b.legal_moves)))
 e.new_game();e.set_position(b.fen())
 packed=np.full((1,32),768,np.int64)
 for j,(sq,pc) in enumerate(b.piece_map().items()):
  packed[0,j]=((pc.piece_type-1)+(0 if pc.color else 6))*64+sq
 first,second,base,out=features(packed,np.array([0 if b.turn else 1]),np.array([[b.king(True),b.king(False)]]),'sage')
 def accumulator(indices):
  indices=indices[indices!=6144]
  return np.clip(e.w1[-1].astype(np.int64)+e.w1[indices].astype(np.int64).sum(axis=0),0,QA)
 a=accumulator(first[0]);c=accumulator(second[0]);head=e.w2.reshape(8,513)[out[0]].astype(np.int64)
 total=np.dot(a*a,head[:256])+np.dot(c*c,head[256:512])+head[-1]*QA*QA
 expected=min(20000,max(-20000,int(base[0])+(int(total)*CP_SCALE)//(QA*QA*QB)))
 actual=ze.evaluate(e.bd,e.st,e.mode,e.tbl_mg,e.tbl_eg,e.acc,0,e.w1,e.w2,e.b2)
 assert expected==actual,(b.fen(),expected,actual)
report['independent_training_contract_evaluations']=100

# Illegal (pinned) en passant must not alter repetition identity.
e.new_game();e.set_position('k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1');h1=zb.compute_hash(e.bd,e.st)
e.set_position('k3r3/8/8/3pP3/8/8/8/4K3 w - - 0 1');assert h1==zb.compute_hash(e.bd,e.st)
report['legal_ep_hash_regression']='passed'

# Threefold vs twofold; no fabricated repetition after a null.
hs=np.array([11,22,11,22,11,22,0,0],np.int64)
assert not zs.is_repetition(hs,2,0,2)
assert zs.is_repetition(hs,4,0,4)
assert not zs.is_repetition(hs,4,0,0)
for ms in (0,10,79,80,100,200,500,1000,3000,120000):
 soft,hard=zengine.budget(ms)
 assert 0<=soft<=hard<ms if ms>0 else soft==hard==0
report['draw_and_budget_regressions']='passed'

# All mate-in-one answers are independently checked with python-chess.
for fen in ('7k/5Q2/6K1/8/8/8/8/8 w - - 0 1',
            '6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1'):
 b=chess.Board(fen);assert b.is_valid()
 e.new_game();uci,score,depth=e.search(fen,300,700,max_depth=8)
 assert chess.Move.from_uci(uci) in b.legal_moves
 b.push_uci(uci);assert b.is_checkmate(),(fen,uci,score,depth)
report['mate_in_one']=2
report['seconds_after_import']=round(time.perf_counter()-START,3)
print(json.dumps(report,indent=2),flush=True)
if '--output' in sys.argv:
 Path(sys.argv[sys.argv.index('--output')+1]).write_text(json.dumps(report,indent=2)+'\n')
