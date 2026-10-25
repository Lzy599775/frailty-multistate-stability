"""Read-only training support prerequisite before registered heldout fits."""
import numpy as np
import study_v8 as s

def rank(df,frame,kind):
    args=s.arrays(df,frame,kind);mat,par,names=s.mapping(frame,kind);design=[]
    for n in range(len(df)):
        for k in np.flatnonzero(args[1][n]>0):
            xx=args[0][n].copy();band=args[2][n,k]
            if band>0:xx[band]=1
            design.append(xx)
    design=np.asarray(design);free=np.vstack([design@mat[k*len(names):(k+1)*len(names)] for k in range(9)])
    return int(np.linalg.matrix_rank(free)),len(par)

def main():
    rows=[]
    for frame in s.FRAMES:
        df=s.old.sample(frame);train,test,_=s.cv_frames(df,frame)
        assert not set(train.person_id)&set(test.person_id)
        for kind in ['original','compact']:
            tr,p=rank(train,frame,kind);expected=p if kind=='compact' or frame=='CHARLS' else 207
            rows.append(dict(frame=frame,model=kind,train_people=train.person_id.nunique(),validation_people=test.person_id.nunique(),train_intervals=len(train),validation_intervals=len(test),parameters=p,training_free_design_rank=tr,expected_supported_rank=expected,extra_rank_loss=expected-tr,absent_observed_endpoint_pairs='|'.join(f'{i}->{j}' for i,j in s.TRANS if not (train.i.eq(i)&train.j.eq(j)).any()),status='PASS' if tr==expected else 'FAIL',qualification_scope='No extra training structural rank loss; rare/absent observed pairs alone do not count latent jumps'))
    s.save(s.OUT/'cv_design_precheck.csv',rows);status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL';s.write(s.OUT/'cv_design_precheck.json',dict(utc=s.utc(),status=status,checks=len(rows),train_fits_started_at_check=any(p.name.startswith(('00_','07_','08_','15_','16_','23_','36_','43_')) for p in (s.OUT/'attempts').iterdir()),no_split_change=True,rows=rows))
    print('CV_DESIGN_PRECHECK',status,rows,flush=True)
    if status!='PASS':raise RuntimeError('STOP training structural support prerequisite')

if __name__=='__main__':main()
