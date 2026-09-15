"""Exercise the actual agent API, default memory configuration and fallbacks."""
import os
if hasattr(os,'sched_getaffinity'):
 os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
import sys,time,json,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
start=time.perf_counter()
import agent,chess
init=time.perf_counter()-start
assert init<90,init
rss=(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // (1024 if sys.platform == "darwin" else 1))
assert rss<1_800_000,rss
rows=[]
positions=[chess.STARTING_FEN,
'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',
'4k3/P7/8/8/8/8/7p/4K3 w - - 0 1',
'k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1',
'8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1']
for i,ms in enumerate([79,100,200,500,1000,3000,120000]):
 fen=positions[i%len(positions)];agent._LAST_REQUEST=None;agent.ENGINE.new_game();agent._SEEN.clear()
 start=time.perf_counter();move=agent.get_move(fen,ms);elapsed=1000*(time.perf_counter()-start)
 assert chess.Move.from_uci(move) in chess.Board(fen).legal_moves
 assert elapsed<ms,(ms,elapsed)
 # Every API path, including low-clock fallback, records our own position.
 assert len(agent.ENGINE.game_hashes)==2,len(agent.ENGINE.game_hashes)
 rows.append({'clock_ms':ms,'elapsed_ms':round(elapsed,3),'move':move,'root_moves_examined':int(agent.ENGINE.sinfo[7])})
 print(json.dumps(rows[-1]),flush=True)
# Fault injection: Python exceptions and malformed proposals must return legal moves.
original=agent.ENGINE.search
for failure in ('exception','illegal'):
 agent._LAST_REQUEST=None;agent.ENGINE.new_game();agent._SEEN.clear()
 def broken(*a,**k):
  if failure=='exception':raise RuntimeError('test injected failure')
  return 'a1a8',0,0
 agent.ENGINE.search=broken
 move=agent.get_move(chess.STARTING_FEN,1000)
 assert chess.Move.from_uci(move) in chess.Board().legal_moves
agent.ENGINE.search=original
# Persistent-game API smoke: both colours separately, legal opponent moves.
for colour in (True,False):
 b=chess.Board();agent.ENGINE.new_game();agent._SEEN.clear();agent._LAST_REQUEST=None
 for ply in range(60):
  if b.is_game_over(claim_draw=True):break
  if b.turn==colour:
   move=chess.Move.from_uci(agent.get_move(b.fen(),500))
  else:
   moves=list(b.legal_moves);move=moves[(ply*17+3)%len(moves)]
  assert move in b.legal_moves;b.push(move)
report={'init_seconds':round(init,3),'peak_rss_kib':(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // (1024 if sys.platform == "darwin" else 1)),
 'tt_bytes':agent.ENGINE.tt_key.nbytes+agent.ENGINE.tt_data.nbytes,
 'clock_cases':rows,'fault_injection_cases':2,'api_smoke_colours':2,
 'python':sys.version,'numpy':__import__('numpy').__version__,'numba':__import__('numba').__version__,
 'chess':chess.__version__,'cpu_affinity':list(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None}
print(json.dumps(report,indent=2),flush=True)
if '--output' in sys.argv:Path(sys.argv[sys.argv.index('--output')+1]).write_text(json.dumps(report,indent=2)+'\n')
