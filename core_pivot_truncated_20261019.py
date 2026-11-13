"""Additive CORE-PIVOT adapter: native objective/search, frozen truncated BC."""
import torch,math
from algorithms.core_pivot import COREPIVOT
from algorithms.truncated_bc import TruncatedPolicy
from algorithms.networks import TwinQ, frozen_copy, optimize, soft_update

class DensityProposal(TruncatedPolicy):
    def sample(self, observations, generator):
        d = self.distribution(observations)
        # Same inverse-CDF distribution as trained BC, portable CPU private RNG.
        u = torch.rand(d.location.shape, dtype=torch.float64,
                       device=generator.device, generator=generator).to(d.location.device)
        eps = torch.finfo(torch.float64).eps
        a = (d.location + d.scale * torch.special.ndtri(
            (d.lower + d.mass*u).clamp(eps, 1-eps))).to(d.dtype)
        if not torch.isfinite(a).all() or (a.abs()>1).any():
            raise ValueError('Invalid bounded proposal sample')
        return a, d.log_prob(a).sum(-1, keepdim=True)

class TruncatedCOREPIVOT(COREPIVOT):
    def _build(self, proposal):
        if not isinstance(proposal, DensityProposal):
            raise TypeError('Verified truncated density proposal required')
        if min(self.config.samples,self.config.bc_candidates,self.config.uniform_candidates)<1 or self.config.conservative_temperature<=0:
            raise ValueError('Invalid proposal/search settings')
        self.proposal=frozen_copy(proposal).to(self.device)
        self.generator=torch.Generator().manual_seed(self.generator.initial_seed())
        self.critic=TwinQ(self.mean,self.scale,self.config.hidden_size).to(self.device)
        self.target=frozen_copy(self.critic)
        self.optimizer=torch.optim.Adam(self.critic.parameters(),lr=self.config.learning_rate)
        self.modules=dict(critic=self.critic,target=self.target,proposal=self.proposal)
        self.optimizers=dict(critic=self.optimizer)

    def penalty(self,b):
        n=len(b['observations']);m=self.config.samples;t=self.config.conservative_temperature
        x=b['observations'][:,None,:].expand(-1,m,-1).reshape(-1,25)
        nx=b['next_observations'][:,None,:].expand(-1,m,-1).reshape(-1,25)
        with torch.no_grad():
            a,lp=self.proposal.sample(x,self.generator);na,nlp=self.proposal.sample(nx,self.generator)
            u=(2*torch.rand((len(x),3),generator=self.generator)-1).to(self.device)
        heads=[[],[]]
        for actions,density in ((u,torch.full_like(lp,-3*math.log(2))),(a,lp),(na,nlp)):
            for head,q in zip(heads,self.critic(x,actions)):head.append((q-density).reshape(n,m))
        return sum((t*torch.logsumexp(torch.cat(head,1)/t,1)-t*math.log(3*m)).mean()-q.mean()
                   for head,q in zip(heads,self.critic(b['observations'],b['actions'])))

    def update(self,b):
        if torch.any(b['timeouts'] & ~b['terminals']):raise ValueError('SARSA timeout mismatch')
        with torch.no_grad():target=b['rewards']+self.config.discount*b['bootstrap_mask']*self.target.minimum(b['next_observations'],b['next_actions'])
        first,second=self.critic(b['observations'],b['actions'])
        bell=(first-target).square().mean()+(second-target).square().mean()
        penalty=self.penalty(b);loss=bell+self.config.conservative_weight*penalty
        optimize(self.optimizer,loss);soft_update(self.critic,self.target,self.config.tau);self.step+=1
        return dict(step=self.step,critic_loss=float(loss.detach()),bellman_loss=float(bell.detach()),conservative_loss=float(penalty.detach()))

    @torch.no_grad()
    def _anchor(self,x):
        a,_=self.proposal.sample(x[None].expand(self.config.bc_candidates,-1),self.generator)
        u=(2*torch.rand((self.config.uniform_candidates,3),generator=self.generator)-1).to(self.device)
        prev=(x[12:15]/5).clamp(-1,1)[None]
        actions=torch.cat((a,u,prev,torch.zeros_like(prev)))
        q=self.critic.minimum(x[None].expand(len(actions),-1),actions)
        return actions[q.argmax()].cpu().numpy()

    def _proposal_state(self):
        return dict(kind='native_truncated_bc',hidden_size=self.proposal.hidden_size,
                    model=self.proposal.state_dict())

    @classmethod
    def _load_proposal(cls,state):
        if state['kind']!='native_truncated_bc':raise ValueError('Wrong proposal kind')
        p=DensityProposal(state['model']['normalize.mean'],state['model']['normalize.scale'],state['hidden_size'])
        p.load_state_dict(state['model']);return p
