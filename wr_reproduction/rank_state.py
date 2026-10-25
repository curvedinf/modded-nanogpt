"""Rank-local activation FP8 delayed scales for eight original microbatches.

Weight caches/scales are common because every rank reads identical weights.
They are refreshed only once per original update, not once per microbatch.
"""
class RankStates:
    def __init__(self,model,ranks=8):
        self.model=model
        self.names=[n for n,b in model.named_buffers() if n.startswith(('_mlp_','_moe_')) and
                    any(s in n for s in ('post_amax','dpre_amax','post_scales','dpre_scales','g_amax'))]
        self.states=[{n:model.get_buffer(n).clone() for n in self.names} for _ in range(ranks)]
    def enter(self,rank):
        for n,value in self.states[rank].items():self.model.get_buffer(n).copy_(value)
    def leave(self,rank):
        for n in self.names:self.states[rank][n].copy_(self.model.get_buffer(n))
    def refresh_activations(self):
        from track_1_short.model.gpt import FP8_POST_HEADROOM,FP8_DPRE_HEADROOM
        for prefix in ('_mlp','_moe'):
            if not hasattr(self.model,prefix+'_post_amax'):continue
            for name,gain,floor in [('post',FP8_POST_HEADROOM/448.,1e-6),('dpre',FP8_DPRE_HEADROOM/57344.,1e-12)]:
                self.model._refresh_delayed_scale(getattr(self.model,prefix+'_'+name+'_amax'),
                    getattr(self.model,prefix+'_'+name+'_scales'),gain,floor)
