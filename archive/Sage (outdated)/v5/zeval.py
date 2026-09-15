"""Original residual NNUE adapter. Deferred feature deltas; direct integer head.

The board/search infrastructure is adapted from the user's supplied Fable.
No third-party engine source is included or translated.
"""
import numpy as np
import os
from numba import njit
from zboard import (SQ120,SQ64,EMPTY,ST_SIDE,ST_KSQ_W,ST_KSQ_B,WHITE,
                    M_FROM,M_TO,M_PROMO,F_EP,F_CASTLE,PIECE_TYPE)
from sl_spec import MG,EG,PHASE,QA,QB,CP_SCALE
CACHE_CHAINS = os.environ.get('SAGE_STAGE3_CACHE_CHAINS','1') == '1'
TBL_MG=np.zeros((13,64),np.int32)
TBL_EG=np.zeros((13,64),np.int32)
for p in range(1,13):
    TBL_MG[p]=MG[(p-1)*64:p*64]
    TBL_EG[p]=EG[(p-1)*64:p*64]

@njit(cache=False)
def region(sq,nb):
    return 0 if nb==1 else 2*((sq//8)//2)+(sq%8)//4

@njit(cache=False)
def feature(p,sq,side,bucket):
    if side==1:
        p=((p-1+6)%12)+1; sq^=56
    return bucket*768+(p-1)*64+sq

@njit(cache=False)
def rebuild_side(bd,acc,ply,side,w1):
    h=w1.shape[1]; bucket=acc[ply,side,h+1]
    for k in range(h): acc[ply,side,k]=w1[-1,k]
    for sq in range(64):
        p=bd[SQ120[sq]]
        if p!=EMPTY:
            idx=feature(p,sq,side,bucket)
            for k in range(h): acc[ply,side,k]+=w1[idx,k]
    acc[ply,side,h]=1

@njit(cache=False)
def nn_refresh(bd,acc,ply,w1,b1):
    h=w1.shape[1]; nb=(w1.shape[0]-1)//768
    mg=0; eg=0; phase=0; count=0
    for sq in range(64):
        p=bd[SQ120[sq]]
        if p:
            mg+=MG[(p-1)*64+sq]; eg+=EG[(p-1)*64+sq]; phase+=PHASE[p-1]; count+=1
            if p==6: acc[ply,0,h+1]=region(sq,nb)
            elif p==12: acc[ply,1,h+1]=region(sq^56,nb)
    acc[ply,0,h+12]=mg; acc[ply,0,h+13]=eg; acc[ply,0,h+14]=phase; acc[ply,0,h+15]=count
    for side in range(2):
        acc[ply,side,h+2]=0; acc[ply,side,h+3]=0
        rebuild_side(bd,acc,ply,side,w1)

@njit(cache=False)
def record_delta(acc,child,h,p,sq,sign):
    for side in range(2):
        n=acc[child,side,h+3]
        acc[child,side,h+4+n]=feature(p,sq,side,acc[child,side,h+1])
        acc[child,side,h+8+n]=sign
        acc[child,side,h+3]=n+1
    idx=(p-1)*64+sq
    acc[child,0,h+12]+=sign*MG[idx]; acc[child,0,h+13]+=sign*EG[idx]
    acc[child,0,h+14]+=sign*PHASE[p-1]; acc[child,0,h+15]+=sign

@njit(cache=False)
def nn_push(bd,st,m,acc,ply,w1):
    # Called before make: only record O(1) metadata, no H-wide neural update.
    h=w1.shape[1]; child=ply+1; nb=(w1.shape[0]-1)//768
    fr=m&M_FROM; to=(m&M_TO)>>7; p=bd[fr]; promo=(m&M_PROMO)>>14
    for side in range(2):
        acc[child,side,h]=0; acc[child,side,h+2]=0; acc[child,side,h+3]=0
        bucket=acc[ply,side,h+1]
        if (p==6 and side==0) or (p==12 and side==1):
            sq=SQ64[to] if side==0 else SQ64[to]^56
            newbucket=region(sq,nb); acc[child,side,h+2]=int(newbucket!=bucket); bucket=newbucket
        acc[child,side,h+1]=bucket
    for k in range(12,16): acc[child,0,h+k]=acc[ply,0,h+k]
    record_delta(acc,child,h,p,SQ64[fr],-1)
    captured=bd[to]
    if m&F_EP:
        capsq=to-10 if p==1 else to+10
        record_delta(acc,child,h,bd[capsq],SQ64[capsq],-1)
    elif captured:
        record_delta(acc,child,h,captured,SQ64[to],-1)
    promoted=promo+(6 if p>6 else 0) if promo else p
    record_delta(acc,child,h,promoted,SQ64[to],1)
    if m&F_CASTLE:
        rf=fr+3 if to>fr else fr-4; rt=fr+1 if to>fr else fr-1
        record_delta(acc,child,h,bd[rf],SQ64[rf],-1)
        record_delta(acc,child,h,bd[rf],SQ64[rt],1)

@njit(cache=False)
def ensure(bd,acc,ply,side,w1):
    h=w1.shape[1]
    if acc[ply,side,h]: return
    ancestor=ply; refresh=False
    while ancestor>0 and not acc[ancestor,side,h]:
        if acc[ancestor,side,h+2]: refresh=True
        ancestor-=1
    if refresh:
        rebuild_side(bd,acc,ply,side,w1); return
    if CACHE_CHAINS:
        # All regions in this chain match, so each ancestor can be reconstructed
        # from feature deltas without restoring its board. Cache it for siblings.
        for node in range(ancestor+1,ply+1):
            for k in range(h): acc[node,side,k]=acc[node-1,side,k]
            for d in range(acc[node,side,h+3]):
                idx=acc[node,side,h+4+d]; sign=acc[node,side,h+8+d]
                for k in range(h): acc[node,side,k]+=sign*w1[idx,k]
            acc[node,side,h]=1
        return
    for k in range(h): acc[ply,side,k]=acc[ancestor,side,k]
    for node in range(ancestor+1,ply+1):
        for d in range(acc[node,side,h+3]):
            idx=acc[node,side,h+4+d]; sign=acc[node,side,h+8+d]
            for k in range(h): acc[ply,side,k]+=sign*w1[idx,k]
    acc[ply,side,h]=1

@njit(cache=False)
def evaluate(bd,st,mode,tbl_mg,tbl_eg,acc,ply,w1,w2,b2):
    h=w1.shape[1]
    ensure(bd,acc,ply,0,w1); ensure(bd,acc,ply,1,w1)
    count=acc[ply,0,h+15]; phase=min(24,acc[ply,0,h+14])
    base=(acc[ply,0,h+12]*phase+acc[ply,0,h+13]*(24-phase))//24
    side=st[ST_SIDE]
    if side: base=-base
    bucket=min(7,max(0,(count-2)//4)); offset=bucket*(2*h+1)
    total=np.int64(w2[offset+2*h])*QA*QA
    for k in range(h):
        a=np.int64(min(QA,max(0,acc[ply,side,k])))
        b=np.int64(min(QA,max(0,acc[ply,side^1,k])))
        total+=a*a*np.int64(w2[offset+k])+b*b*np.int64(w2[offset+h+k])
    result=base+(total*CP_SCALE)//(QA*QA*QB)
    return min(20000,max(-20000,result))

def dummy_nn():
    return np.zeros((769,256),np.int16),np.zeros(256,np.int16),np.zeros(8*513,np.int16),np.int32(0)
