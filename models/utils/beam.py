import torch

class GNMTGlobalScorer(object):
    """
    NMT re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`
    Args:
       alpha (float): length parameter
       beta (float):  coverage parameter
    """

    def __init__(self):
      super().__init__()

    def score(self, beam, logprobs):
        # """
        # Rescores a prediction based on penalty functions
        # """
        # normalized_probs = self.length_penalty(beam,
        #                                        logprobs,
        #                                        self.alpha)
        # if not beam.stepwise_penalty:
        #     penalty = self.cov_penalty(beam,
        #                                beam.global_state["coverage"],
        #                                self.beta)
        #     normalized_probs -= penalty

        return logprobs


class Beam(object):
  def __init__(self, size, pad, bos, eos, global_scorer, min_length):
    super().__init__()
    self.prev_ks = []
    self.scores = torch.FloatTensor(size).zero_()
    self.all_scores = []

    self.min_length = min_length

    self.next_ys = [torch.LongTensor(size)
                        .fill_(pad)]
    self.next_ys[0][0] = bos

    self.finished = []

    self.global_scorer = global_scorer

    self._eos = eos
    self.eos_top = False

    self.size = size

  def get_current_state(self):
    "Get the outputs for the current timestep."
    return self.next_ys[-1]

  def get_current_origin(self):
    "Get the backpointers for the current timestep."
    return self.prev_ks[-1]
  def done(self):
    return self.eos_top
  def sort_finished(self, minimum=None):
    if minimum is not None:
        i = 0
        # Add from beam until we have minimum outputs.
        while len(self.finished) < minimum:
            global_scores = self.global_scorer.score(self, self.scores)
            s = global_scores[i]
            self.finished.append((s, len(self.next_ys) - 1, i))
            i += 1

    self.finished.sort(key=lambda a: -a[0])
    scores = [sc for sc, _, _ in self.finished]
    ks = [(t, k) for _, t, k in self.finished]
    return scores, ks

  def get_hyp(self, timestep, k):
      """
      Walk back to construct the full hypothesis.
      """
      hyp = []
      for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
          hyp.append(self.next_ys[j + 1][k])
          k = self.prev_ks[j][k]
      return hyp[::-1]
  def advance(self, word_probs):
    word_probs = word_probs.unsqueeze(0)
    # print("word_probs", word_probs.shape)
    num_words = word_probs.size(2)
    
    # dont let this shit end
    cur_len = len(self.next_ys)
    # print("cur_len", cur_len)
    if cur_len < self.min_length:
        for k in range(len(word_probs)):
            word_probs[0][k][self._eos] = -1e20
            # print("word", self._eos, "has", word_probs[k][self._eos])

    if len(self.prev_ks) > 0:
      beam_scores = word_probs + self.scores.unsqueeze(1).expand_as(word_probs)
      # print("beam_scores.shape", beam_scores.shape, "word_probs", word_probs.shape, "self.scores.unsqueeze(1).expand_as(word_probs)", self.scores.unsqueeze(1).expand_as(word_probs).shape)
      beam_scores = beam_scores.squeeze(0)
      # print("beam_scores", beam_scores)
      for i in range(self.next_ys[-1].size(0)):
        if self.next_ys[-1][i] == self._eos:
          # print("sentence", i, 'is ended')
          beam_scores[i] = -1e20
    else:
      beam_scores = word_probs[0][0]
      # print("beam_scores.shape first time", beam_scores.shape)

    
    flat_beam_scores = beam_scores.view(-1)
    # print("self.size", self.size, "flat_beam_scores.shape", flat_beam_scores.shape)
    
    best_scores, best_scores_id = flat_beam_scores.topk(self.size, 0,
        True, True)
    
    self.all_scores.append(self.scores)
    self.scores = best_scores
    # print("best_scores", best_scores, "best_scores_id", best_scores_id)
    
    # print("flat_beam_scores", flat_beam_scores, "best_scores", best_scores, "best_scores_id", best_scores_id)
    # print("num_words", num_words)
    prev_k = torch.floor(best_scores_id / num_words).long()
    # print("prev_k", prev_k)
    self.prev_ks.append(prev_k)
    self.next_ys.append((best_scores_id - prev_k * num_words))
    # print("self.prev_ks", self.prev_ks)
    # print("self.next_ys", self.next_ys)
    for i in range(self.next_ys[-1].size(0)):
      if self.next_ys[-1][i] == self._eos:
        # print("self.scores", self.scores)
        global_scores = self.global_scorer.score(self, self.scores)
        s = global_scores[i]
        self.finished.append((s, len(self.next_ys) - 1, i))

    if self.next_ys[-1][0] == self._eos:
      self.all_scores.append(self.scores)
      self.eos_top = True
        
def _from_beam(beam):
    ret = {"predictions": [],
            "scores": [],
            "attention": []}
    for b in beam:
        # n_best = self.n_best
        scores, ks = b.sort_finished(minimum=None)
        # print(scores, ks)
        hyps, attn = [], []
        for i, (times, k) in enumerate(ks):#[:n_best]):
            # hyp, att = b.get_hyp(times, k)
            hyp = b.get_hyp(times, k)
            hyps.append(hyp)
            # attn.append(att)
        ret["predictions"].append(hyps)
        ret["scores"].append(scores)
        # ret["attention"].append(attn)
    return ret