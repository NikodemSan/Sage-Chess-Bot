"""Reproducible local matches. No external engine is shipped or required.

python tests/match.py --opponent /path/to/extracted/other/engine --games 8
Each worker gets one CPU and is idle between moves; state is reset per game.
Use --fresh-workers to additionally recreate each process between games.
Default is a short regression clock. --seconds 120 --increment 0.5 matches the
competition clock, but local CPU hardware still differs from the match host.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

WORKER = r'''
import os,sys,time,json,resource
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['NUMBA_NUM_THREADS']='1'
os.environ['SAGE_TT_BITS']=os.environ.get('MATCH_TT_BITS','25' if os.environ.get('MATCH_API') == '1' else '21')
if hasattr(os,'sched_getaffinity'):
    os.sched_setaffinity(0,{int(os.environ.get('MATCH_CPU',min(os.sched_getaffinity(0))))})
root,model,weightpath=sys.argv[1:4]
sys.path.insert(0,root)
t=time.perf_counter()
import chess,zengine
api_mode = os.environ.get('MATCH_API') == '1'
if api_mode:
    import importlib
    os.environ['SAGE_MODEL']=model
    name='agent' if (__import__('pathlib').Path(root)/'agent.py').exists() else 'reference_agent'
    api=importlib.import_module(name)
    engine=api.ENGINE
else:
    engine=zengine.Engine(weightpath)
    if model=='ensemble': engine.blend_with(str(__import__('pathlib').Path(root)/'models/sage50m.npz'))
    for fen in [chess.STARTING_FEN,'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1']:
        engine.search(fen,10,20,max_depth=3)
    engine.push_own_move(int(engine.last_move))
print(json.dumps({'ready':True,'init_s':time.perf_counter()-t,'rss_kib':(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // (1024 if sys.platform == "darwin" else 1))}),flush=True)
for line in sys.stdin:
    req=json.loads(line)
    if req.get('reset'):
        engine.new_game()
        if api_mode and hasattr(api,'_LAST_REQUEST'):api._LAST_REQUEST=None
        if api_mode and hasattr(api,'_SEEN'):api._SEEN.clear()
        print('{}',flush=True);continue
    t=time.perf_counter()
    fen=req['fen'];left=req['left'];inc=req['inc']
    if api_mode:
        uci=api.get_move(fen,int(left))
        score=int(engine.sinfo[4]);depth=int(getattr(engine,'last_completed_depth',0))
    else:
        if req.get('fixed_ms'):
            soft=hard=req['fixed_ms']
        else:
            soft,hard=zengine.budget(left,inc,chess.Board(fen).ply())
        uci,score,depth=engine.search(fen,soft,hard)
        engine.push_own_move(engine.last_move)
    print(json.dumps({'move':uci,'score':int(score),'depth':depth,'nodes':int(engine.sinfo[0]),'ms':1000*(time.perf_counter()-t),'rss_kib':(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // (1024 if sys.platform == "darwin" else 1))}),flush=True)
'''

OPENINGS = [
[], ['e2e4','e7e5','g1f3','b8c6','f1b5','a7a6','b5a4','g8f6'],
['d2d4','d7d5','c2c4','e7e6','b1c3','g8f6','c1g5','f8e7'],
['e2e4','c7c5','g1f3','d7d6','d2d4','c5d4','f3d4','g8f6','b1c3','a7a6'],
['d2d4','g8f6','c2c4','g7g6','b1c3','f8g7','e2e4','d7d6'],
['c2c4','e7e5','b1c3','g8f6','g2g3','d7d5','c4d5','f6d5'],
]

class Worker:
 def __init__(self,root,model):
  root=Path(root).resolve()
  weights=root/'models'/('sage50m.npz' if model=='50m' else 'sage150m.npz')
  if not weights.exists(): weights=root/'weights.npz'
  self.p=subprocess.Popen([sys.executable,'-u','-c',WORKER,str(root),model,str(weights)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=sys.stderr,text=True)
  self.ready=self.read(90)
 def read(self,timeout):
  import selectors
  with selectors.DefaultSelector() as sel:
   sel.register(self.p.stdout,selectors.EVENT_READ)
   if not sel.select(timeout):
    self.p.kill();raise TimeoutError('Worker missed deadline')
  line=self.p.stdout.readline()
  if not line: raise RuntimeError('Worker exited')
  return json.loads(line)
 def call(self,msg,timeout=10):
  self.p.stdin.write(json.dumps(msg)+'\n');self.p.stdin.flush();return self.read(timeout)
 def close(self):
  self.p.terminate()
  try:self.p.wait(timeout=3)
  except subprocess.TimeoutExpired:self.p.kill();self.p.wait()

def run(args):
 import chess,chess.pgn
 root=Path(__file__).resolve().parents[1]
 if args.api:os.environ['MATCH_API']='1'
 a=Worker(root,args.model);b=Worker(args.opponent,args.opponent_model)
 print(json.dumps({'init':[a.ready,b.ready]}),flush=True)
 results=[];games=[];totals={'candidate_wins':0,'draws':0,'opponent_wins':0}
 try:
  for game in range(args.games):
   if game and args.fresh_workers:
    a.close();b.close();a=Worker(root,args.model);b=Worker(args.opponent,args.opponent_model)
   a.call({'reset':True});b.call({'reset':True})
   board=chess.Board()
   for uci in OPENINGS[(game//2)%len(OPENINGS)]:board.push_uci(uci)
   initial=board.fen();record=chess.pgn.Game();record.setup(board);node=record
   candidate_white=game%2==0
   record.headers['White']='candidate' if candidate_white else 'opponent'
   record.headers['Black']='opponent' if candidate_white else 'candidate'
   record.headers['TimeControl']=f'{args.seconds}+{args.increment}'
   clocks={True:args.seconds*1000,False:args.seconds*1000}
   failure=None;log=[]
   while board.ply()<600 and len(log)<args.ply_cap and not board.is_game_over(claim_draw=True):
    side=board.turn;worker=a if side==candidate_white else b
    started=time.perf_counter()
    try:
     response=worker.call({'fen':board.fen(),'left':clocks[side],'inc':args.increment*1000,'fixed_ms':args.fixed_ms},timeout=max(0.1,clocks[side]/1000))
     elapsed=1000*(time.perf_counter()-started)
     clocks[side]-=elapsed
     if clocks[side]<=0:failure='flag';break
     move=chess.Move.from_uci(response['move'])
     if move not in board.legal_moves:failure='illegal';break
    except TimeoutError:failure='flag';break
    except Exception as exc:failure=repr(exc);break
    log.append(dict(response,side='white' if side else 'black',clock_ms=clocks[side]))
    board.push(move);node=node.add_variation(move);clocks[side]+=args.increment*1000
    if len(log)%20==0:
     print(json.dumps({'game':game+1,'plies':len(log),'white_ms':round(clocks[True]),'black_ms':round(clocks[False]),'last_move':move.uci()}),flush=True)
   if failure:
    result='1/2-1/2' if failure=='flag' and board.has_insufficient_material(not board.turn) else ('0-1' if board.turn else '1-0')
   else:result=board.result(claim_draw=True)
   if result=='*':result='1/2-1/2'
   record.headers['Result']=result
   if result=='1/2-1/2':totals['draws']+=1
   elif (result=='1-0')==candidate_white:totals['candidate_wins']+=1
   else:totals['opponent_wins']+=1
   item={'game':game+1,'candidate_white':candidate_white,'result':result,'plies':len(log),'failure':failure,'capped':len(log)>=args.ply_cap,'clocks_ms':clocks,'moves':log,'initial_fen':initial}
   results.append(item);games.append(str(record))
   print(json.dumps({k:v for k,v in item.items() if k not in ('moves','initial_fen')}|totals),flush=True)
  output={'settings':vars(args),'init':[a.ready,b.ready],'totals':totals,'games':results,'note':'Small local regression match; no reliable Elo or tournament-rank inference.'}
  Path(args.output).write_text(json.dumps(output,indent=2)+'\n')
  Path(args.output).with_suffix('.pgn').write_text('\n\n'.join(games)+'\n')
 finally:a.close();b.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--opponent',required=True);p.add_argument('--api',action='store_true');p.add_argument('--fresh-workers',action='store_true');p.add_argument('--model',default='150m',choices=['150m','50m','ensemble']);p.add_argument('--opponent-model',default='150m',choices=['150m','50m','ensemble']);p.add_argument('--games',type=int,default=8);p.add_argument('--seconds',type=float,default=20);p.add_argument('--increment',type=float,default=0.1);p.add_argument('--fixed-ms',type=float,default=0);p.add_argument('--ply-cap',type=int,default=600);p.add_argument('--output',default='match_results.json');run(p.parse_args())
