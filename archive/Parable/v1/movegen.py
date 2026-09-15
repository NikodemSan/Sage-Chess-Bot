import numpy as np
from numba import njit

WHITE=1; BLACK=-1
P=1; N=2; B=3; R=4; Q=5; K=6
WK=1; WQ=2; BK=4; BQ=8
FLAG_NORMAL=0; FLAG_EP=1; FLAG_CASTLE=2; FLAG_PAWN2=3
MATE=30000
MAX_PLY=96

@njit(cache=False, inline='always')
def enc_move(fr,to,promo=0,flag=0):
    return np.int32(fr | (to<<6) | (promo<<12) | (flag<<15))
@njit(cache=False, inline='always')
def m_from(m): return m & 63
@njit(cache=False, inline='always')
def m_to(m): return (m>>6)&63
@njit(cache=False, inline='always')
def m_promo(m): return (m>>12)&7
@njit(cache=False, inline='always')
def m_flag(m): return (m>>15)&7

@njit(cache=False)
def find_king(board, side):
    target=side*K
    for s in range(64):
        if board[s]==target: return s
    return -1

@njit(cache=False)
def is_attacked(board, sq, by_side):
    sf=sq&7; sr=sq>>3
    if by_side==WHITE:
        s=sq-7
        if s>=0 and abs((s&7)-sf)==1 and board[s]==P: return True
        s=sq-9
        if s>=0 and abs((s&7)-sf)==1 and board[s]==P: return True
    else:
        s=sq+7
        if s<64 and abs((s&7)-sf)==1 and board[s]==-P: return True
        s=sq+9
        if s<64 and abs((s&7)-sf)==1 and board[s]==-P: return True
    for dr,df in ((2,1),(2,-1),(1,2),(1,-2),(-1,2),(-1,-2),(-2,1),(-2,-1)):
        r=sr+dr; f=sf+df
        if 0<=r<8 and 0<=f<8 and board[r*8+f]==by_side*N: return True
    for dr in range(-1,2):
        for df in range(-1,2):
            if dr==0 and df==0: continue
            r=sr+dr; f=sf+df
            if 0<=r<8 and 0<=f<8 and board[r*8+f]==by_side*K: return True
    for dr,df in ((1,0),(-1,0),(0,1),(0,-1)):
        r=sr+dr; f=sf+df
        while 0<=r<8 and 0<=f<8:
            pc=board[r*8+f]
            if pc:
                if pc==by_side*R or pc==by_side*Q: return True
                break
            r+=dr; f+=df
    for dr,df in ((1,1),(1,-1),(-1,1),(-1,-1)):
        r=sr+dr; f=sf+df
        while 0<=r<8 and 0<=f<8:
            pc=board[r*8+f]
            if pc:
                if pc==by_side*B or pc==by_side*Q: return True
                break
            r+=dr; f+=df
    return False

@njit(cache=False, inline='always')
def add_move(moves,n,fr,to,promo=0,flag=0):
    moves[n]=enc_move(fr,to,promo,flag); return n+1

@njit(cache=False)
def gen_pseudo(board,side,rights,ep,moves):
    n=0
    for fr in range(64):
        pc=int(board[fr])
        if pc*side<=0: continue
        pt=abs(pc); r=fr>>3; f=fr&7
        if pt==P:
            dr=1 if side==WHITE else -1
            to=fr+8*dr
            promo_rank=7 if side==WHITE else 0
            start_rank=1 if side==WHITE else 6
            if 0<=to<64 and board[to]==0:
                if (to>>3)==promo_rank:
                    for pr in (Q,R,B,N): n=add_move(moves,n,fr,to,pr)
                else:
                    n=add_move(moves,n,fr,to)
                    to2=fr+16*dr
                    if r==start_rank and board[to2]==0:
                        n=add_move(moves,n,fr,to2,0,FLAG_PAWN2)
            for df in (-1,1):
                tf=f+df
                if not (0<=tf<8): continue
                to=fr+8*dr+df
                if not (0<=to<64): continue
                if board[to]*side<0:
                    if (to>>3)==promo_rank:
                        for pr in (Q,R,B,N): n=add_move(moves,n,fr,to,pr)
                    else: n=add_move(moves,n,fr,to)
                elif to==ep:
                    n=add_move(moves,n,fr,to,0,FLAG_EP)
        elif pt==N:
            for dr,df in ((2,1),(2,-1),(1,2),(1,-2),(-1,2),(-1,-2),(-2,1),(-2,-1)):
                tr=r+dr; tf=f+df
                if 0<=tr<8 and 0<=tf<8:
                    to=tr*8+tf
                    if board[to]*side<=0: n=add_move(moves,n,fr,to)
        elif pt in (B,R,Q):
            if pt in (B,Q):
                for dr,df in ((1,1),(1,-1),(-1,1),(-1,-1)):
                    tr=r+dr; tf=f+df
                    while 0<=tr<8 and 0<=tf<8:
                        to=tr*8+tf; dst=board[to]
                        if dst==0: n=add_move(moves,n,fr,to)
                        else:
                            if dst*side<0: n=add_move(moves,n,fr,to)
                            break
                        tr+=dr; tf+=df
            if pt in (R,Q):
                for dr,df in ((1,0),(-1,0),(0,1),(0,-1)):
                    tr=r+dr; tf=f+df
                    while 0<=tr<8 and 0<=tf<8:
                        to=tr*8+tf; dst=board[to]
                        if dst==0: n=add_move(moves,n,fr,to)
                        else:
                            if dst*side<0: n=add_move(moves,n,fr,to)
                            break
                        tr+=dr; tf+=df
        else:
            for dr in range(-1,2):
                for df in range(-1,2):
                    if dr==0 and df==0: continue
                    tr=r+dr; tf=f+df
                    if 0<=tr<8 and 0<=tf<8:
                        to=tr*8+tf
                        if board[to]*side<=0: n=add_move(moves,n,fr,to)
            if side==WHITE and fr==4:
                if (rights&WK) and board[5]==0 and board[6]==0 and board[7]==R:
                    if not is_attacked(board,4,BLACK) and not is_attacked(board,5,BLACK) and not is_attacked(board,6,BLACK):
                        n=add_move(moves,n,4,6,0,FLAG_CASTLE)
                if (rights&WQ) and board[3]==0 and board[2]==0 and board[1]==0 and board[0]==R:
                    if not is_attacked(board,4,BLACK) and not is_attacked(board,3,BLACK) and not is_attacked(board,2,BLACK):
                        n=add_move(moves,n,4,2,0,FLAG_CASTLE)
            elif side==BLACK and fr==60:
                if (rights&BK) and board[61]==0 and board[62]==0 and board[63]==-R:
                    if not is_attacked(board,60,WHITE) and not is_attacked(board,61,WHITE) and not is_attacked(board,62,WHITE):
                        n=add_move(moves,n,60,62,0,FLAG_CASTLE)
                if (rights&BQ) and board[59]==0 and board[58]==0 and board[57]==0 and board[56]==-R:
                    if not is_attacked(board,60,WHITE) and not is_attacked(board,59,WHITE) and not is_attacked(board,58,WHITE):
                        n=add_move(moves,n,60,58,0,FLAG_CASTLE)
    return n

@njit(cache=False)
def update_rights(rights,fr,to,moving,captured):
    if moving==K: rights &= ~(WK|WQ)
    elif moving==-K: rights &= ~(BK|BQ)
    elif moving==R:
        if fr==0: rights &= ~WQ
        elif fr==7: rights &= ~WK
    elif moving==-R:
        if fr==56: rights &= ~BQ
        elif fr==63: rights &= ~BK
    if captured==R:
        if to==0: rights &= ~WQ
        elif to==7: rights &= ~WK
    elif captured==-R:
        if to==56: rights &= ~BQ
        elif to==63: rights &= ~BK
    return rights

@njit(cache=False)
def make_move(board,m,side,rights,ep):
    fr=m_from(m); to=m_to(m); pr=m_promo(m); fl=m_flag(m)
    moving=int(board[fr]); captured=int(board[to])
    board[fr]=0
    if fl==FLAG_EP:
        cs=to-8*side; captured=int(board[cs]); board[cs]=0
    if fl==FLAG_CASTLE:
        if to==6: board[5]=board[7]; board[7]=0
        elif to==2: board[3]=board[0]; board[0]=0
        elif to==62: board[61]=board[63]; board[63]=0
        elif to==58: board[59]=board[56]; board[56]=0
    board[to]=side*pr if pr else moving
    nr=update_rights(rights,fr,to,moving,captured)
    nep=(fr+to)//2 if fl==FLAG_PAWN2 else -1
    return captured,nr,nep

@njit(cache=False)
def unmake_move(board,m,side,captured):
    fr=m_from(m); to=m_to(m); pr=m_promo(m); fl=m_flag(m)
    moved=int(board[to])
    if fl==FLAG_CASTLE:
        if to==6: board[7]=board[5]; board[5]=0
        elif to==2: board[0]=board[3]; board[3]=0
        elif to==62: board[63]=board[61]; board[61]=0
        elif to==58: board[56]=board[59]; board[59]=0
    if fl==FLAG_EP:
        board[fr]=side*P; board[to]=0; board[to-8*side]=captured
    else:
        board[fr]=side*P if pr else moved; board[to]=captured

@njit(cache=False)
def gen_legal(board,side,rights,ep,moves):
    n=gen_pseudo(board,side,rights,ep,moves)
    king0=find_king(board,side); out=0
    for i in range(n):
        m=moves[i]; fr=m_from(m); to=m_to(m); moving=int(board[fr])
        cap,nr,nep=make_move(board,m,side,rights,ep)
        ks=to if abs(moving)==K else king0
        legal=ks>=0 and not is_attacked(board,ks,-side)
        unmake_move(board,m,side,cap)
        if legal: moves[out]=m; out+=1
    return out

@njit(cache=False)
def perft(board,side,rights,ep,depth,buffers,ply=0):
    if depth==0: return 1
    moves=buffers[ply]; n=gen_legal(board,side,rights,ep,moves)
    if depth==1: return n
    total=0
    for i in range(n):
        m=moves[i]; cap,nr,nep=make_move(board,m,side,rights,ep)
        total += perft(board,-side,nr,nep,depth-1,buffers,ply+1)
        unmake_move(board,m,side,cap)
    return total

def parse_fen(fen):
    parts=fen.split(); board=np.zeros(64,dtype=np.int8)
    mp={'P':P,'N':N,'B':B,'R':R,'Q':Q,'K':K,'p':-P,'n':-N,'b':-B,'r':-R,'q':-Q,'k':-K}
    for ri,row in enumerate(parts[0].split('/')):
        rank=7-ri; file=0
        for c in row:
            if c.isdigit(): file+=int(c)
            else: board[rank*8+file]=mp[c]; file+=1
    side=WHITE if parts[1]=='w' else BLACK
    rights=0; cr=parts[2]
    if 'K' in cr: rights|=WK
    if 'Q' in cr: rights|=WQ
    if 'k' in cr: rights|=BK
    if 'q' in cr: rights|=BQ
    ep=-1
    if parts[3]!='-': ep=(ord(parts[3][0])-97)+8*(int(parts[3][1])-1)
    halfmove=int(parts[4]) if len(parts)>4 else 0
    fullmove=int(parts[5]) if len(parts)>5 else 1
    return board,side,rights,ep,halfmove,fullmove

def move_to_uci(m):
    fr=int(m)&63; to=(int(m)>>6)&63; pr=(int(m)>>12)&7
    s=chr(97+(fr&7))+str((fr>>3)+1)+chr(97+(to&7))+str((to>>3)+1)
    if pr: s+={N:'n',B:'b',R:'r',Q:'q'}[pr]
    return s
