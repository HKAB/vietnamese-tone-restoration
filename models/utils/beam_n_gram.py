import re

def remove_vn_accent(word):
    word = re.sub('[áàảãạăắằẳẵặâấầẩẫậ]', 'a', word)
    word = re.sub('[éèẻẽẹêếềểễệ]', 'e', word)
    word = re.sub('[óòỏõọôốồổỗộơớờởỡợ]', 'o', word)
    word = re.sub('[íìỉĩị]', 'i', word)
    word = re.sub('[úùủũụưứừửữự]', 'u', word)
    word = re.sub('[ýỳỷỹỵ]', 'y', word)
    word = re.sub('đ', 'd', word)
    return word

def gen_accents_word(word):
    word_no_accent = remove_vn_accent(word.lower())
    all_accent_word = {word}
    for w in open('data/vn_syllables.txt').read().splitlines():
        w_no_accent = remove_vn_accent(w.lower())
        if w_no_accent == word_no_accent:
            all_accent_word.add(w)
    return all_accent_word

# beam search
def beam_search_n_gram(words, model, k=3):
  sequences = []
  for idx, word in enumerate(words):
    if idx == 0:
      sequences = [([x], 0.0) for x in gen_accents_word(word)]
    else:
      all_sequences = []
      for seq in sequences:
        for next_word in gen_accents_word(word):
          current_word = seq[0][-1]
          try:
              previous_word = seq[0][-2]
              score = model.logscore(next_word, [previous_word, current_word])
          except:
              score = model.logscore(next_word, [current_word])
          new_seq = seq[0].copy()
          new_seq.append(next_word)
          all_sequences.append((new_seq, seq[1] + score))
      all_sequences = sorted(all_sequences,key=lambda x: x[1], reverse=True)
      sequences = all_sequences[:k]
  return sequences