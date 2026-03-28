from typing import List, Tuple
import collections
import heapq
import math
import kenlm
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


class Wav2Vec2Decoder:
    def __init__(
            self,
            model_name="facebook/wav2vec2-base-100h",
            lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
            beam_width=3,
            alpha=1.0,
            beta=1.0,
            temperature=1.0,
        ):
        """
        Initialization of Wav2Vec2Decoder class
        
        Args:
            model_name (str): Pretrained Wav2Vec2 model from transformers
            lm_model_path (str): Path to the KenLM n-gram model (for LM rescoring)
            beam_width (int): Number of hypotheses to keep in beam search
            alpha (float): LM weight for shallow fusion and rescoring
            beta (float): Word bonus for shallow fusion
        """
        # once logits are available, no other interactions with the model are allowed
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        # you can interact with these parameters
        self.vocab = {i: c for c, i in self.processor.tokenizer.get_vocab().items()}
        self.blank_token_id = self.processor.tokenizer.pad_token_id
        self.word_delimiter = self.processor.tokenizer.word_delimiter_token
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.lm_model = kenlm.Model(lm_model_path) if lm_model_path else None

    def _select_top_beams(self, scored_hypotheses):
        top_hypotheses = []

        for idx, (score, hypothesis) in enumerate(scored_hypotheses):
            entry = (score, idx, hypothesis)
            if len(top_hypotheses) < self.beam_width:
                heapq.heappush(top_hypotheses, entry)
            else:
                heapq.heappushpop(top_hypotheses, entry)

        top_hypotheses.sort(key=lambda x: x[0], reverse=True)
        return [hypothesis for _, _, hypothesis in top_hypotheses]
        
    def greedy_decode(self, scores: torch.Tensor) -> str:
        """
        Perform greedy decoding (find best CTC path)
        
        Args:
            scores (torch.Tensor): Acoustic model outputs with shape (T, V) or (1, T, V)
        
        Returns:
            str: Decoded transcript
        """
        if scores.ndim == 3:
            scores = scores[0]

        predicted_ids = torch.argmax(scores, dim=-1).tolist()
        return self._decode_path(predicted_ids)

    def beam_search_decode(self, scores: torch.Tensor, return_beams: bool = False):
        """
        Perform beam search decoding (no LM)
        
        Args:
            scores (torch.Tensor): Log probabilities from Wav2Vec2 model (T, V) or (1, T, V),
                where T - number of time steps and V - vocabulary size
            return_beams (bool): Return all beam hypotheses for second pass LM rescoring
        
        Returns:
            Union[str, List[Tuple[float, List[int]]]]: 
                (str) - If return_beams is False, returns the best decoded transcript as a string.
                (List[Tuple[List[int], float]]) - If return_beams is True, returns a list of tuples
                    containing hypotheses and log probabilities.
        """
        NEG_INF = -float("inf")

        def make_new_beam():
            return collections.defaultdict(lambda: (NEG_INF, NEG_INF))

        def logsumexp(*args):
            if all(a == NEG_INF for a in args):
                return NEG_INF
            a_max = max(args)
            return a_max + math.log(sum(math.exp(a - a_max) for a in args))

        log_probs = scores[0] if scores.ndim == 3 else scores
        T, S = log_probs.shape

        beam = [(tuple(), (0.0, NEG_INF))]

        for t in range(T):
            next_beam = make_new_beam()

            for s in range(S):
                p = log_probs[t, s].item()

                for prefix, (p_b, p_nb) in beam:
                    if s == self.blank_token_id:
                        # пересчитываем вероятности для бланка
                        n_p_b, n_p_nb = next_beam[prefix]
                        n_p_b = logsumexp(n_p_b, p_b + p, p_nb + p)
                        next_beam[prefix] = (n_p_b, n_p_nb)
                        continue

                    # дополняем новым символом
                    end_t = prefix[-1] if prefix else None
                    n_prefix = prefix + (s,)
                    n_p_b, n_p_nb = next_beam[n_prefix]

                    # пересчитываем вероятности
                    if s != end_t:
                        n_p_nb = logsumexp(n_p_nb, p_b + p, p_nb + p)
                    else:
                        n_p_nb = logsumexp(n_p_nb, p_b + p)

                    next_beam[n_prefix] = (n_p_b, n_p_nb)

                    if s == end_t:
                        n_p_b, n_p_nb = next_beam[prefix]
                        n_p_nb = logsumexp(n_p_nb, p_nb + p)
                        next_beam[prefix] = (n_p_b, n_p_nb)

            beam = self._select_top_beams(
                (logsumexp(*scores), (path, scores))
                for path, scores in next_beam.items()
            )

        if return_beams:
            return [(list(path), logsumexp(*scores)) for path, scores in beam]
        else:
            best_prefix, best_scores = beam[0]
            return self._decode_prefix(best_prefix)

    def _decode_path(self, path) -> str:
        string_list = []
        word_list = []

        for i in range(len(path)):
            if path[i] == self.blank_token_id:
                continue
            if i != 0 and path[i] == path[i - 1]:
                continue
            if self.vocab[path[i]] != self.word_delimiter:
                word_list.append(self.vocab[path[i]])
            else:
                string_list.append(''.join(word_list))
                word_list = []

        if word_list:
            string_list.append(''.join(word_list))

        return ' '.join(string_list).lower().strip()

    def _decode_prefix(self, path) -> str:
        string_list = []
        word_list = []

        for c in path:
            if c == self.blank_token_id:
                continue
            if self.vocab[c] != self.word_delimiter:
                word_list.append(self.vocab[c])
            else:
                string_list.append(''.join(word_list))
                word_list = []

        if word_list:
            string_list.append(''.join(word_list))

        return ' '.join(string_list).lower().strip()
    
    def beam_search_with_lm(self, scores: torch.Tensor) -> str:
        if not self.lm_model:
            raise ValueError("KenLM model required for LM shallow fusion")

        NEG_INF = -float("inf")

        def make_new_beam():
            return collections.defaultdict(lambda: (NEG_INF, NEG_INF))

        def logsumexp(*args):
            if all(a == NEG_INF for a in args):
                return NEG_INF
            a_max = max(args)
            return a_max + math.log(sum(math.exp(a - a_max) for a in args))

        text_cache = {}
        lm_score_cache = {}
        word_count_cache = {}

        def get_text(prefix):
            if prefix in text_cache:
                return text_cache[prefix]

            text_cache[prefix] = self._decode_prefix(prefix).lower().strip()
            return text_cache[prefix]

        def get_lm_score(prefix):
            if prefix in lm_score_cache:
                return lm_score_cache[prefix]

            text = get_text(prefix)

            if not text:
                return 0.0

            score = self.lm_model.score(text, bos=True, eos=False) * math.log(10)
            lm_score_cache[prefix] = score
            return score

        def get_word_count(prefix):
            if prefix in word_count_cache:
                return word_count_cache[prefix]

            text = get_text(prefix)

            count = len(text.split()) if text else 0
            word_count_cache[prefix] = count
            return count

        log_probs = scores[0] if scores.ndim == 3 else scores
        T, S = log_probs.shape

        beam = [(tuple(), (0.0, NEG_INF))]

        for t in range(T):
            next_beam = make_new_beam()

            for s in range(S):
                p = log_probs[t, s].item()

                for prefix, (p_b, p_nb) in beam:
                    if s == self.blank_token_id:
                        n_p_b, n_p_nb = next_beam[prefix]
                        n_p_b = logsumexp(n_p_b, p_b + p, p_nb + p)
                        next_beam[prefix] = (n_p_b, n_p_nb)
                        continue

                    end_t = prefix[-1] if prefix else None
                    n_prefix = prefix + (s,)
                    n_p_b, n_p_nb = next_beam[n_prefix]

                    if s != end_t:
                        n_p_nb = logsumexp(n_p_nb, p_b + p, p_nb + p)
                    else:
                        n_p_nb = logsumexp(n_p_nb, p_b + p)

                    next_beam[n_prefix] = (n_p_b, n_p_nb)

                    if s == end_t:
                        n_p_b, n_p_nb = next_beam[prefix]
                        n_p_nb = logsumexp(n_p_nb, p_nb + p)
                        next_beam[prefix] = (n_p_b, n_p_nb)

            beam = self._select_top_beams(
                (
                    logsumexp(*scores) + self.alpha * get_lm_score(path) + self.beta * get_word_count(path),
                    (path, scores),
                )
                for path, scores in next_beam.items()
            )

        best_prefix, _ = beam[0]
        return self._decode_prefix(best_prefix)


    def lm_rescore(self, beams: List[Tuple[List[int], float]]) -> str:
        """
        Perform second-pass LM rescoring on beam search outputs
        
        Args:
            beams (list): List of tuples (hypothesis, log_prob)
        
        Returns:
            str: Best rescored transcript
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM rescoring")
        best_score = -float("inf")
        best_text = ""

        for path, acoustic_score in beams:
            text = self._decode_prefix(path).lower().strip()
            if not text:
                continue
            lm_score = self.lm_model.score(text, bos=True, eos=False) * math.log(10)
            word_count = len(text.split())
            total = acoustic_score + self.alpha * lm_score + self.beta * word_count

            if total > best_score:
                best_score = total
                best_text = text

        return best_text

    def decode(self, audio_input: torch.Tensor, method: str = "greedy") -> str:
        """
        Decode input audio file using the specified method
        
        Args:
            audio_input (torch.Tensor): Audio tensor
            method (str): Decoding method ("greedy", "beam", "beam_lm", "beam_lm_rescore"),
                where "greedy" is a greedy decoding,
                      "beam" is beam search without LM,
                      "beam_lm" is beam search with LM shallow fusion, and 
                      "beam_lm_rescore" is a beam search with second pass LM rescoring
        
        Returns:
            str: Decoded transcription
        """
        if isinstance(audio_input, torch.Tensor):
            audio_input = audio_input.squeeze().cpu().numpy()

        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            logits = logits / self.temperature
            log_probs = torch.log_softmax(logits, dim=-1)

        if method == "greedy":
            return self.greedy_decode(log_probs)
        elif method == "beam":
            return self.beam_search_decode(log_probs)
        elif method == "beam_lm":
            return self.beam_search_with_lm(log_probs)
        elif method == "beam_lm_rescore":
            beams = self.beam_search_decode(log_probs, return_beams=True)
            return self.lm_rescore(beams)
        else:
            raise ValueError("Invalid decoding method. Choose one of 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'.")


def test(decoder, audio_path, true_transcription):

    import Levenshtein

    audio_input, sr = torchaudio.load(audio_path)
    assert sr == 16000, "Audio sample rate must be 16kHz"

    print("=" * 60)
    print("Target transcription")
    print(true_transcription)

    # Print all decoding methods results
    for d_strategy in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
        print("-" * 60)
        print(f"{d_strategy} decoding") 
        transcript = decoder.decode(audio_input, method=d_strategy)
        print(f"{transcript}")
        print(f"Character-level Levenshtein distance: {Levenshtein.distance(true_transcription, transcript.strip())}")


if __name__ == "__main__":
    
    test_samples = [
        ("examples/sample1.wav", "IF YOU ARE GENEROUS HERE IS A FITTING OPPORTUNITY FOR THE EXERCISE OF YOUR MAGNANIMITY IF YOU ARE PROUD HERE AM I YOUR RIVAL READY TO ACKNOWLEDGE MYSELF YOUR DEBTOR FOR AN ACT OF THE MOST NOBLE FORBEARANCE"),
        ("examples/sample2.wav", "AND IF ANY OF THE OTHER COPS HAD PRIVATE RACKETS OF THEIR OWN IZZY WAS UNDOUBTEDLY THE MAN TO FIND IT OUT AND USE THE INFORMATION WITH A BEAT SUCH AS THAT EVEN GOING HALVES AND WITH ALL THE GRAFT TO THE UPPER BRACKETS HE'D STILL BE ABLE TO MAKE HIS PILE IN A MATTER OF MONTHS"),
        ("examples/sample3.wav", "GUESS A MAN GETS USED TO ANYTHING HELL MAYBE I CAN HIRE SOME BUMS TO SIT AROUND AND WHOOP IT UP WHEN THE SHIPS COME IN AND BILL THIS AS A REAL OLD MARTIAN DEN OF SIN"),
        ("examples/sample4.wav", "IT WAS A TUNE THEY HAD ALL HEARD HUNDREDS OF TIMES SO THERE WAS NO DIFFICULTY IN TURNING OUT A PASSABLE IMITATION OF IT TO THE IMPROVISED STRAINS OF I DIDN'T WANT TO DO IT THE PRISONER STRODE FORTH TO FREEDOM"),
        ("examples/sample5.wav", "MARGUERITE TIRED OUT WITH THIS LONG CONFESSION THREW HERSELF BACK ON THE SOFA AND TO STIFLE A SLIGHT COUGH PUT UP HER HANDKERCHIEF TO HER LIPS AND FROM THAT TO HER EYES"),
        ("examples/sample6.wav", "AT THIS TIME ALL PARTICIPANTS ARE IN A LISTEN ONLY MODE"),
        ("examples/sample7.wav", "THE INCREASE WAS MAINLY ATTRIBUTABLE TO THE NET INCREASE IN THE AVERAGE SIZE OF OUR FLEETS"),
        ("examples/sample8.wav", "OPERATING SURPLUS IS A NON CAP FINANCIAL MEASURE WHICH IS DEFINED AS FULLY IN OUR PRESS RELEASE"),
    ]

    decoder = Wav2Vec2Decoder()

    _ = [test(decoder, audio_path, target) for audio_path, target in test_samples]
