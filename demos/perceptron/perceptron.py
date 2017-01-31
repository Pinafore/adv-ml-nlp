import operator
import argparse
from collections import defaultdict
from math import log

from nltk import bigrams

from numpy import zeros

kSTART = "___START___"
kNEG_INF = float("-inf")


class ToyDataset:
    def __init__(self):
        self._sents = ["answer the question".split(),
                       "question the answer".split(),
                       "you demand the delay".split(),
                       "you delay the demand".split(),
                       "what silence can show".split(),
                       "what show can silence".split()]

        self._tags = ["VB DET NN".split(),
                      "VB DET NN".split(),
                      "PRO VB DET NN".split(),
                      "PRO VB DET NN".split(),
                      "PRO NN VB VB".split(),
                      "PRO NN VB VB".split()]

    def tagged_sents(self):
        for sent, tag in zip(self._sents, self._tags):
            yield zip(sent, tag)

    def words(self):
        for ii in self._sents:
            for jj in ii:
                yield jj

    def tagged_words(self):
        for sent, tag in zip(self._sents, self._tags):
            for ii, jj in zip(sent, tag):
                yield ii, jj


class HmmDataset(ToyDataset):
    def __init__(self):
        self._sents = ["a blue boat".split(),
                       "the old man".split(),
                       "the old man the boat".split(),
                       "an old man".split()]

        self._tags = ["D A N V".split(),
                      "D A N".split(),
                      "D A V D N".split(),
                      "D A N".split()]

class ExamDataset(ToyDataset):
    def __init__(self):
        self._sents = ["time flies like an arrow".split(),
                       "fruit flies like an apple".split()]
        self._tags = ["N V P D N".split(),
                      "A N V D N".split()]

class SlideFunctions:
    def __init__(self, tagging_perceptron):
        self._tp = tagging_perceptron
        self._tags = tagging_perceptron._tags

    def slide_prediction(self, sent_num, sent, pred, score, bp):
        val = []

        val.append("\\begin{frame}{Decoding Sentence %i}\n\n\n" % sent_num)

        # Table cells for later
        table = {}
        display_num = 1
        for ii, ww in enumerate(sent):
            for tt, tag in enumerate(self._tags):
                display_num += 1
                table[(tt, -1)] = "%s\t&" % tag
                table[(tt, ii)] = "\\only<%i->{\\alert<%i>{%0.2f}}\t&" % \
                    (display_num, display_num, score[ii][tag])

                val.append("\\only<%i>{$" % display_num)
                if ii == 0:
                    prev_tag = kSTART
                else:
                    prev_tag = bp[ii][tag]
                    val.append("\\delta_%i(%s) + " % (ii - 1, prev_tag))

                val.append("w_{\\mbox{%s, %s}} + w_{\\mbox{%s, %s}} = " %
                           (prev_tag.replace("_", ""), tag, tag, ww))

                if prev_tag != kSTART:
                    val.append("%0.2f + " % score[ii - 1][prev_tag])
                val.append("%0.2f + %0.2f = \\alert<%i>{%0.2f}" %
                           (self._tp((prev_tag, tag)), self._tp((tag, ww)),
                            display_num, score[ii][tag]))

                val.append("$}\n")
                table[(tt, len(sent))] = "\\cr\n"

        val.append("\\begin{itemize}\n")
        val.append("\t\\item Scores\n\n")

        val.append("\\begin{equation}\n")
        val.append("\\delta = \\bordermatrix{&")
        val.append("\t&".join("\\mbox{%s}_{%i}" % (ww, ss)
                              for ss, ww in enumerate(sent)))
        val.append("\\cr\n")
        for ii in sorted(table):
            val.append(table[ii])
        val.append("}\\end{equation}\n\n")

        display_num += 1
        val.append("\\only<%i->{\n" % display_num)
        val.append("\\item Backpointers\n")

        val.append("\\begin{equation}\n")
        val.append("\\beta = \\bordermatrix{&")
        val.append("\t&".join("\\mbox{%s}_{%i}" % (sent[x], x)
                              for x in bp))
        val.append("\\cr\n")

        display_num += 1
        for tt, tag in enumerate(self._tags):
            val.append("\\mbox{%s}" % tag)
            for ii in bp:
                if bp[ii][tag] == pred[ii - 1] and pred[ii] == tag:
                    val.append("&\t\\alert<%i>{%s}" %
                               (display_num, bp[ii][tag]))
                else:
                    val.append("&\t%s" % bp[ii][tag])
            val.append("\\cr\n")
        val.append("}\\end{equation}\n")
        val.append("}\n")

        display_num += 1
        val.append("\\only<%i->{\n" % display_num)
        val.append("\\item Reconstruction: ")
        val.append(" ".join(pred))
        val.append("}\n")

        val.append("\\end{itemize}\n")

        val.append("\\end{frame}\n%-----------------------\n\n\n")
        return "".join(val)

    def slide_update(self, old_weight_vector, new_weight_vector,
                     sent, pred, gold):
        val = []
        val.append("\\begin{frame}\n")
        val.append("\\begin{itemize}\n")

        val.append("\\item Correct answer: " + " ".join(gold) + "\n")
        val.append("\\item Prediction:")
        for ii, pp in enumerate(pred):
            if gold[ii] == pp:
                val.append(" %s" % pp)
            else:
                val.append(" \\alert<2>{%s}" % pp)

        gold_feat = self._tp.feature_vector(sent, gold, True)
        pred_feat = self._tp.feature_vector(sent, pred, True)

        val.append("\\only<3->{")
        val.append("\\begin{columns}\n")
        val.append("\\column{.3\\linewidth}\n")
        val.append("\\begin{block}{Gold Features}\n")
        for ii in gold_feat:
            if not ii in pred_feat:
                val.append("%s " % str(ii).replace(" ", "~"))
        val.append("\\end{block}\n")

        val.append("\\column{.3\\linewidth}\n")
        val.append("\\begin{block}{Shared Features}\n")
        for ii in gold_feat:
            if ii in pred_feat:
                val.append("%s " % str(ii).replace(" ", "~"))
        val.append("\\end{block}\n")

        val.append("\\column{.3\\linewidth}\n")
        val.append("\\begin{block}{Predicted Features}\n")
        for ii in pred_feat:
            if not ii in gold_feat:
                val.append("%s " % str(ii).replace(" ", "~"))
        val.append("\\end{block}\n")
        val.append("\\end{columns}\n")
        val.append("}")
        val.append("\\only<4->{\n")
        val.append("\\item New feature vector: ")
        for ii in sorted(new_weight_vector):
            if new_weight_vector[ii] != 0.0 or not ii in old_weight_vector or \
                    old_weight_vector[ii] != 0.0:
                if not ii in old_weight_vector or \
                        new_weight_vector[ii] != old_weight_vector[ii]:
                    val.append("\\alert<5>{%s}:~%0.2f; " %
                               (str(ii).replace(" ", "~"), new_weight_vector[ii]))
                else:
                    val.append("%s: %0.2f; " % (str(ii), new_weight_vector[ii]))
        # Remove last semi-colon
        val[-1] = val[-1][:-2]
        val.append("}\n")
        val.append("\\end{itemize}\n")
        val.append("\\end{frame}\n%-----------------------\n\n\n")

        return "".join(val).replace("'", "").replace("_", "")


# Utility functions
def argmax(stats):
    assert len(stats) > 0, "Array needs to be non-empty %s" % str(stats)
    return max(stats.iteritems(), key=operator.itemgetter(1))[0]


def normalize(word):
    return word.lower()


def vocabulary(corpus):
    return set(normalize(x) for x in corpus.words())


def tags(corpus):
    return set(x[1] for x in corpus.tagged_words())


class TaggingPerceptron:
    def __init__(self, vocabulary, tag_set, average=True):
        self._tags = list(tag_set)
        self._voc = list(vocabulary)
        self._n = 0
        self._final = False

        self._feature_ids = {}

        self._num_feats = 0
        for ii in self._tags + [kSTART]:
            for jj in self._tags:
                self._feature_ids[(ii, jj)] = self._num_feats
                self._num_feats += 1

        for word in vocabulary:
            for tag in tag_set:
                self._feature_ids[(tag, word)] = self._num_feats
                self._num_feats += 1
        self._w = zeros(self._num_feats)
        self._sum = zeros(self._num_feats)

    def __call__(self, feat):
        assert feat in self._feature_ids, \
            "%s feature not found" % str(feat)
        return self._w[self._feature_ids[feat]]

    def decode(self, sentence):
        scores = defaultdict(dict)
        backpointers = defaultdict(dict)

        # start probabilities
        for ii in self._tags:
            scores[0][ii] = self((ii, sentence[0])) + \
                self((kSTART, ii))

        for ii, ww in enumerate(sentence[1:], 1):
            for tt in self._tags:
                # Find the best previous score
                previous = dict((x, scores[ii - 1][x] + self((x, tt)))
                                for x in self._tags)
                word_score = self((tt, ww))
                best = argmax(previous)
                scores[ii][tt] = word_score + previous[best]
                backpointers[ii][tt] = best
                # print(ii, tt, scores)

        # reconstruct
        tag_sequence = [argmax(scores[len(sentence) - 1])]
        for ii in xrange(len(sentence) - 1, 0, -1):
            tag_sequence.insert(0, backpointers[ii][tag_sequence[0]])

        return tag_sequence, scores, backpointers

    def feature_vector(self, sentence, tags, create_dictionary=False):
        if not create_dictionary:
            f = zeros(self._num_feats)
            for ii, jj in bigrams([kSTART] + tags):
                f[self._feature_ids[(ii, jj)]] += 1.0

            for tt, ww in zip(tags, sentence):
                f[self._feature_ids[(tt, ww)]] += 1.0
        else:
            f = defaultdict(float)
            for ii, jj in bigrams([kSTART] + tags):
                f[(ii, jj)] += 1.0

            for tt, ww in zip(tags, sentence):
                f[(tt, ww)] += 1.0

        return f

    def finalize(self):
        """
        We're done with learning, set the weight parameter to the averaged
        perceptron weight
        """
        self._final = True
        self._w = self._sum / float(self._n)

    def update(self, sentence, predicted, gold):
        assert not self._final, "Cannot update once we've finalized weights"
        diff = self.feature_vector(sentence, gold) -\
            self.feature_vector(sentence, predicted)
        self._w += diff

        self._sum += self._w
        self._n += 1

        return self._w, diff

    def accuracy(self, test_sents, test_tags):
        total = 0
        right = 0
        confusion = defaultdict(dict)
        word = defaultdict(dict)

        for ss, tt in zip(test_sents, test_tags):
            pred, scores, backpointers = self.decode(ss)

            for ww, pp, gg in zip(ss, pred, tt):
                total += 1
                if pp != gg:
                    confusion[pp][gg] = confusion[pp].get(gg, 0) + 1
                    word[(ww, pp)][gg] = word[(ww, pp)].get(gg, 0) + 1
                else:
                    right += 1
        return float(right) / float(total), confusion, word

    def pretty_weights(self, weight_vector=None):
        """
        Create a vector with real features as keys of dictionary
        """

        if weight_vector is None:
            weight_vector = self._w
        d = {}

        for ii in self._feature_ids:
            val = weight_vector[self._feature_ids[ii]]
            if val != 0.0:
                d[ii] = val
        return d

    def train(self, iters, train_sents, train_tags, test_sents, test_tags,
              report_every=100):

        for ii in xrange(iters):
            for ss, tt in zip(train_sents, train_tags):
                print(ss)
                print(tt)
                pred, score, backpointers = self.decode(ss)
                assert len(pred) == len(ss), "Pred tag len mismatch"
                assert len(ss) == len(tt), "Gold tag len mismatch"
                print(pred)
                self.update(ss, pred, tt)

                if self._n % report_every == 1:
                    test_accuracy, confusion, word_errors = \
                        self.accuracy(test_sents, test_tags)
                    print("\t".join(ss))
                    print("\t".join(tt))
                    print("\t".join(pred))
                    print("After %i sents, accuracy is %f, nonzero feats %i" %
                          (self._n, test_accuracy,
                           sum(1 for x in self._w if x != 0.0)))

        self.finalize()
        test_accuracy, confusion, word_errors = \
            self.accuracy(test_sents, test_tags)
        print("---------------")
        print("Final accuracy: %f" % test_accuracy)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", help="Which dataset to use",
                           type=str, default='brown', required=False)
    argparser.add_argument("--iterations", help="How many passes over data",
                           type=int, default=10, required=False)
    argparser.add_argument("--test_size", help="Size of test set (in sents)",
                           type=int, default=500, required=False)
    argparser.add_argument("--demo", help="Write LaTeX output for demo",
                           type=str, default="", required=False)
    flags = argparser.parse_args()
    if flags.dataset == 'brown':
        from nltk.corpus import brown
        dataset = brown
    elif flags.dataset == 'treebank':
        from nltk.corpus import treebank
        dataset = treebank
    elif flags.dataset == 'hmm':
        dataset = HmmDataset()
    elif flags.dataset == 'exam':
        dataset = ExamDataset()
    else:
        dataset = ToyDataset()

    print(tags(dataset))
    tp = TaggingPerceptron(vocabulary(dataset),
                           tags(dataset))

    if flags.test_size > 0:
        train_sents = [(lambda x: [y[0] for y in x])(pair) for
                       pair in dataset.tagged_sents()[:-flags.test_size]]
        train_tags = [(lambda x: [y[1] for y in x])(pair) for
                      pair in dataset.tagged_sents()[:-flags.test_size]]
        test_sents = [(lambda x: [y[0] for y in x])(pair) for
                      pair in dataset.tagged_sents()[-flags.test_size:]]
        test_tags = [(lambda x: [y[1] for y in x])(pair) for
                     pair in dataset.tagged_sents()[-flags.test_size:]]
    else:
        train_sents = [(lambda x: [y[0] for y in x])(pair) for
                       pair in dataset.tagged_sents()]
        train_tags = [(lambda x: [y[1] for y in x])(pair) for
                      pair in dataset.tagged_sents()]
        test_sents = train_sents
        test_tags = train_tags

    if flags.demo == "hmm.tex":
        sf = SlideFunctions(tp)
        o = open(flags.demo, 'w')
        sentence = 0
        print(tp._feature_ids)
        tp._w[tp._feature_ids[(kSTART, "D")]] = log(0.3, 10)
        tp._w[tp._feature_ids[(kSTART, "A")]] = log(0.3, 10)
        tp._w[tp._feature_ids[(kSTART, "N")]] = log(0.3, 10)
        tp._w[tp._feature_ids[(kSTART, "V")]] = log(0.1, 10)

        tp._w[tp._feature_ids[("D", "D")]] = log(0.1, 10)
        tp._w[tp._feature_ids[("D", "A")]] = log(0.4, 10)
        tp._w[tp._feature_ids[("D", "N")]] = log(0.45, 10)
        tp._w[tp._feature_ids[("D", "V")]] = log(0.05, 10)

        tp._w[tp._feature_ids[("A", "D")]] = log(0.1, 10)
        tp._w[tp._feature_ids[("A", "A")]] = log(0.3, 10)
        tp._w[tp._feature_ids[("A", "N")]] = log(0.5, 10)
        tp._w[tp._feature_ids[("A", "V")]] = log(0.1, 10)

        tp._w[tp._feature_ids[("N", "D")]] = log(0.05, 10)
        tp._w[tp._feature_ids[("N", "A")]] = log(0.05, 10)
        tp._w[tp._feature_ids[("N", "N")]] = log(0.1, 10)
        tp._w[tp._feature_ids[("N", "V")]] = log(0.8, 10)

        tp._w[tp._feature_ids[("V", "D")]] = log(0.3, 10)
        tp._w[tp._feature_ids[("V", "A")]] = log(0.2, 10)
        tp._w[tp._feature_ids[("V", "N")]] = log(0.3, 10)
        tp._w[tp._feature_ids[("V", "V")]] = log(0.2, 10)

        tp._w[tp._feature_ids[("D", "the")]] = log(0.6, 10)
        tp._w[tp._feature_ids[("D", "old")]] = log(0.025, 10)
        tp._w[tp._feature_ids[("D", "man")]] = log(0.025, 10)
        tp._w[tp._feature_ids[("D", "blue")]] = log(0.025, 10)
        tp._w[tp._feature_ids[("D", "boat")]] = log(0.025, 10)
        tp._w[tp._feature_ids[("D", "a")]] = log(0.2, 10)
        tp._w[tp._feature_ids[("D", "an")]] = log(0.1, 10)

        tp._w[tp._feature_ids[("A", "the")]] = log(0.033, 10)
        tp._w[tp._feature_ids[("A", "old")]] = log(0.3, 10)
        tp._w[tp._feature_ids[("A", "man")]] = log(0.1, 10)
        tp._w[tp._feature_ids[("A", "blue")]] = log(0.3, 10)
        tp._w[tp._feature_ids[("A", "boat")]] = log(0.1, 10)
        tp._w[tp._feature_ids[("A", "a")]] = log(0.033, 10)
        tp._w[tp._feature_ids[("A", "an")]] = log(0.033, 10)

        tp._w[tp._feature_ids[("N", "the")]] = log(0.033, 10)
        tp._w[tp._feature_ids[("N", "old")]] = log(0.1, 10)
        tp._w[tp._feature_ids[("N", "man")]] = log(0.4, 10)
        tp._w[tp._feature_ids[("N", "blue")]] = log(0.1, 10)
        tp._w[tp._feature_ids[("N", "boat")]] = log(0.3, 10)
        tp._w[tp._feature_ids[("N", "a")]] = log(0.033, 10)
        tp._w[tp._feature_ids[("N", "an")]] = log(0.033, 10)

        tp._w[tp._feature_ids[("V", "the")]] = log(0.033, 10)
        tp._w[tp._feature_ids[("V", "old")]] = log(0.1, 10)
        tp._w[tp._feature_ids[("V", "man")]] = log(0.4, 10)
        tp._w[tp._feature_ids[("V", "blue")]] = log(0.2, 10)
        tp._w[tp._feature_ids[("V", "boat")]] = log(0.2, 10)
        tp._w[tp._feature_ids[("V", "a")]] = log(0.033, 10)
        tp._w[tp._feature_ids[("V", "an")]] = log(0.033, 10)

        for ss, tt in zip(train_sents, train_tags):
            sentence += 1

            pred, score, backpointers = tp.decode(ss)
            o.write(sf.slide_prediction(sentence, ss, pred,
                                        score, backpointers))
            old_vector = tp.pretty_weights()

    elif flags.demo != "":
        sf = SlideFunctions(tp)
        o = open(flags.demo, 'w')
        sentence = 0
        for ss, tt in zip(train_sents, train_tags):
            sentence += 1
            pred, score, backpointers = tp.decode(ss)
            o.write(sf.slide_prediction(sentence, ss, pred,
                                        score, backpointers))
            old_vector = tp.pretty_weights()
            w, d = tp.update(ss, pred, tt)
            new_vector = tp.pretty_weights()

            o.write(sf.slide_update(old_vector, new_vector, ss, pred, tt))

    else:
        tp.train(flags.iterations, train_sents, train_tags,
                 test_sents, test_tags)
