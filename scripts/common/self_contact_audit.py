"""Read-only candidate reward-scale audit; never changes env rewards."""
from __future__ import annotations


class SelfContactAudit:
    def __init__(self, dt, weight=-0.1):
        self.dt=float(dt);self.weight=float(weight);self.rows={};self.events=[]

    @staticmethod
    def empty():
        return dict(samples=0,raw_sum=0.,raw_max=0.,nonzero_samples=0,
                    task_reward_sum=0.,peak_force_n=0.,max_duration_s=0.)

    def update(self, env, extras, row_ids, valid, rewards):
        import torch
        sc=extras['self_contact'];p=extras['getup_progress'];monitor=env.scene.sensors['self_contact']
        selected=valid.nonzero().flatten()
        if not len(selected):return
        matrix=torch.stack((row_ids.float(),sc['penalty_raw'],sc['peak_pair_force_n'],
            sc['max_duration_s'],p['succeeded'].float(),p['hold_seconds'],rewards),-1)
        values=matrix[selected].detach().cpu().tolist()
        for e,values_e in zip(selected.tolist(),values):
            row,raw,force,duration,succeeded,hold,task=values_e;key=str(int(row))
            phase='pre_success' if not succeeded else ('post_success_stable' if hold>=3 else 'post_success_unstable')
            entry=self.rows.setdefault(key,{'all':self.empty(),'phases':{}})
            for item in [entry['all'],entry['phases'].setdefault(phase,self.empty())]:
                item['samples']+=1;item['raw_sum']+=raw;item['raw_max']=max(item['raw_max'],raw)
                item['nonzero_samples']+=int(raw>1e-6);item['task_reward_sum']+=task
                item['peak_force_n']=max(item['peak_force_n'],force);item['max_duration_s']=max(item['max_duration_s'],duration)
            # Preserve a few strongest events; the vector cache is cleared on auto reset.
            if force>500 and (len(self.events)<30 or force>min(x['force_n'] for x in self.events)):
                f=monitor.peak_force[e];v,k=f.max(0)
                event=dict(bank_row=int(row),env_id=e,force_n=force,phase=phase,
                    time_s=float(env.episode_length_buf[e]*env.step_dt),
                    pair_force_n=float(v),pair_cache_missing=bool(float(v)<=0.),
                    pair_duration_at_control_end_s=float(monitor.duration[e,k]),
                    penalty_raw=float(sc['penalty_raw'][e]),
                    birth_type=int(p.get('birth_types',row_ids)[e]),
                    bank_state_row=int(p.get('bank_row_ids',row_ids)[e]))
                if float(v)>0:
                    event.update(body_a=monitor.body_names[int(monitor._i[k])],body_b=monitor.body_names[int(monitor._j[k])])
                self.events.append(event);self.events.sort(key=lambda x:x['force_n'],reverse=True);self.events=self.events[:30]

    def report(self):
        import copy
        rows=copy.deepcopy(self.rows)
        for entry in rows.values():
            for item in [entry['all'],*entry['phases'].values()]:
                n=max(1,item['samples']);item['raw_mean']=item['raw_sum']/n
                item['nonzero_fraction']=item['nonzero_samples']/n
                item['virtual_weighted_return']=self.weight*self.dt*item['raw_sum']
                item['virtual_weighted_per_step']=self.weight*self.dt*item['raw_mean']
        return dict(weight_candidate=self.weight,dt=self.dt,rows=rows,top_events=self.events,
            note='Hypothetical reward only; environment rewards unchanged. pre_success includes the required first stable hold.')
