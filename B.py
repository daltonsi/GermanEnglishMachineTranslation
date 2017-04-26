import nltk
import A
from collections import defaultdict
from nltk.align import AlignedSent
from nltk.align import Alignment

class BerkeleyAligner():

    def __init__(self, align_sents, num_iter):
        self.t, self.q = self.train(align_sents, num_iter)

    # TODO: Computes the alignments for align_sent, using this model's parameters. Return
    #       an AlignedSent object, with the sentence pair and the alignments computed.
    def align(self, align_sent):

        best_alignment = []

        # PRO TIP from ISABEL GAO - put all terms in lower case
        german = [word.lower() for word in align_sent.words]
        english = [word.lower() for word in align_sent.mots]



        l = len(english)
        m = len(german)

        # https://piazza.com/class/isgem432wbx5um?cid=327
        # Changed to use mots on the outside loop
        """for i, g_word in enumerate(german):


            best_p = (self.t[g_word][None] * self.q[0][i+1][l][m])
            best_p = max(best_p, 0)
            best_alignment_pt = 0

            for j, e_word in enumerate(english):
                align_p = (self.t[g_word][e_word] * self.q[j+1][i+1][l][m])
                if align_p >= best_p:
                    best_p = align_p
                    best_alignment_pt = j


            best_alignment.append((i, best_alignment_pt))"""

        for j, e_word in enumerate(english):
            best_p = (self.t[None][e_word] * self.q[j+1][0][l][m])
            best_p = max(best_p, 0)
            best_alignment_pt = 0

            for i, g_word in enumerate(german):
                align_p = (self.t[g_word][e_word] * self.q[j+1][i+1][l][m])
                if align_p >= best_p:
                    best_p = align_p
                    best_alignment_pt = i

            best_alignment.append((best_alignment_pt, j))


        final_align_sent = AlignedSent(german,
                                english,
                                Alignment(best_alignment))
        return final_align_sent


    # TODO: Implement the EM algorithm. num_iters is the number of iterations. Returns the
    # translation and distortion parameters as a tuple.
    def train(self, aligned_sents, num_iters):

        #############################
        # Part 0: Preliminaries     #
        #############################

        # Create a set of dictionaries for translation and for distortion parameters
        # One for each model to be trained, the two will be averaged together for the final parameters for alignment
        # Each model proceeds in a different direction
        #       1: mots -> words direction, i.e from english to german
        #       2: words -> mots direction, i.e from german to english

        #translation
        t1 = defaultdict(lambda: defaultdict(lambda:0))
        t2 = defaultdict(lambda: defaultdict(lambda:0))

        #distortion/alignment
        q1 = defaultdict(lambda: 0.0)
        #q2 = defaultdict(lambda: 0.0)
        #q1 = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:defaultdict(lambda:0))))
        #q2 = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:defaultdict(lambda:0))))

        #############################
        # Part 1: EM Initialization #
        #############################

        # Create German and English vocabularies
        ger_vocab = set()  # all unique words
        eng_vocab = set()  # all unque mots

        for sentence in aligned_sents:

            #Alignment options
            # PRO TIP from ISABEL GAO - put all terms in lower case
            german = [word.lower() for word in sentence.words]
            english = [word.lower() for word in sentence.mots]

            #sentence lengths
            m = len(german)
            l = len(english)

            # Update Vocabularies
            ger_vocab.update(german)
            eng_vocab.update(english)

            # Initialize alignment/distortion q parameters
            for i in range(l+1):
                for j in range(m+1):
                    if i !=0 or j != 0:
                        q1[(i,j,m,l)] = 1.0 / (l+1)
                        #q2[(i,j,m,l)] = 1.0 / (m+1)
                        #q1[i][j][m][l] = 1.0 / (l+1)
                        #q2[i][j][m][l] = 1.0 / (m+1)

        # Initialize translation t parameters
        total_ger_vocab = len(ger_vocab)
        total_eng_vocab = len(eng_vocab)

        for g_word in ger_vocab:
            # include null alignments for german
            t1[g_word][None] = 1.0 / (total_ger_vocab+1)
        for e_word in eng_vocab:
            t2[None][e_word] = 1.0 / (total_eng_vocab+1)

        for sentence in aligned_sents:
            german = [word.lower() for word in sentence.words]
            for g_word in german:
                english = [word.lower() for word in sentence.mots]
                for e_word in english:
                    t1[g_word][e_word] = 1.0 / (total_ger_vocab+1)
                    t2[g_word][e_word] = 1.0 / (total_eng_vocab+1)



        #############################
        # Part 2: EM Implementation #
        #############################
        for x in range(num_iters):

            # Pair counts
            ## c(e,g) and c(g,e)
            ge_total = defaultdict(lambda: 0.0) # c(g,e)  used with t1 and q1
            eg_total = defaultdict(lambda: 0.0) # c(e,g)  used with t2 and q2

            # Language counts
            ## c(e) and c(f)
            e_total = defaultdict(lambda: 0.0) # c(e)  used with t1 and q1
            g_total = defaultdict(lambda: 0.0) # c(g)  used with t2 and q2

            pairs = defaultdict(lambda: 0.0)

            # alignment numerators
            # number of times an alignment of i in german is observed with alignment j in german
            # for given sentence lengths of m in german and l in english
            align = defaultdict(lambda: 0.0) # c(j|i,m,l)  used with t1 and q2

            align2 = defaultdict(lambda: 0.0) # c(i|j,m,l)  used with t2 and q2

            # alignment denominators
            ## c(i,l,m) and c(j,m,l)

            # number of times we see an eng sentence of length l with a
            # german sentence of length m for a given german word position i
            #  i = position of german word
            #  l = length of english sentence
            #  m = length of german sentence
            align_g = defaultdict(lambda: 0.0) # c(i,l,m)  used with t1 and q1

            # number of times we see an german sentence of length m with a
            # english sentence of length l for a given english word position j
            #  j = position of english word
            #  l = length of english sentence
            #  m = length of german sentence
            align_e = defaultdict(lambda: 0.0) # c(j,m,l)  used with t2 and q2

            for sentence in aligned_sents:

                german = [None] + [word.lower() for word in sentence.words]
                english = [None] + [word.lower() for word in sentence.mots]

                # sentence lengths
                m = len(german) - 1
                l = len(english) - 1

                # denominators for normalization
                # sums the alignment/translation products for a given g -> e and e -> g pair
                #      j = positions in english
                #      i = positions in german
                for i in range(1,m+1):
                    g_word = german[i]
                    ge_total[g_word] = 0
                    for j in range(l+1):
                        e_word = english[j]
                        ge_total[g_word] += t1[g_word][e_word] * q1[(j,i,m,l)]

                        #ge_total[g_word] += t1[g_word][e_word] * q1[j][i][m][l]

                for j in range(1,l+1):
                    e_word = english[j]
                    eg_total[e_word] = 0
                    for i in range(m+1):
                        g_word = german[i]
                        eg_total[e_word] += t2[g_word][e_word] * q1[(j,i,m,l)]
                        #eg_total[e_word] += t2[g_word][e_word] * q2[j][i][m][l]

                # Counters for each model
                for i in range(m+1):
                    g_word = german[i]
                    for j in range(l+1):
                        if i == 0 and j == 0:
                            continue
                        e_word = english[j]

                        if ge_total[g_word] == 0:
                            delta1 = 0
                        else:
                            delta1 = t1[g_word][e_word] * q1[(j,i,m,l)] / ge_total[g_word]
                            #delta1 = t1[g_word][e_word] * q1[j][i][m][l] / ge_total[g_word]

                        if eg_total[e_word] == 0:
                            delta2 = 0
                        else:
                            delta2 = t2[g_word][e_word] * q1[(j,i,m,l)] / eg_total[e_word]
                            #delta2 = t2[g_word][e_word] * q2[j][i][m][l] / eg_total[e_word]


                        delta = (delta1 + delta2) ** 2 / 2.0

                        e_total[(e_word)] += delta
                        g_total[(g_word)] += delta
                        pairs[(g_word, e_word)] += delta
                        align[(j,i,m,l)] += delta
                        align_g[(i,m,l)] += delta
                        align_e[(j,m,l)] += delta

            #############################
            # Part 2.5: Update params   #
            #############################

            for sentence in aligned_sents:
                german = [None] + [word.lower() for word in sentence.words]
            english = [None] + [word.lower() for word in sentence.mots]

            m = len(german) - 1
            l = len(english) - 1

            for g_word in ger_vocab:
                for e_word in eng_vocab:
                    p = pairs[(g_word,e_word)]
                    if p > 0:
                        if not e_total[e_word] == 0:
                            #t1[(g_word,e_word)] = p * 1.0 / e_total[e_word]
                            t1[g_word][e_word] = p * 1.0 / e_total[e_word]
                        if not g_total[g_word] == 0:
                            #t2[(g_word,e_word)] = p * 1.0 / g_total[g_word]
                            t2[g_word][e_word] = p * 1.0 / g_total[g_word]

            for sentence in aligned_sents:
                german = [None] + [word.lower() for word in sentence.words]
                english = [None] + [word.lower() for word in sentence.mots]
                m = len(german) - 1
                l= len(english) - 1

                for j in range(l + 1):
                    for i in range(m + 1):
                        p = align[(j,i,m,l)]
                        if p > 0:
                            if not align_g[(i,m,l)] == 0:
                                q1[(j,i,m,l)] = p * 1.0 / align_g[(i,m,l)]
                                #q1[j][i][m][l] = p * 1.0 / align_g[(i,m,l)]
                            #if not align_e[(j, m, l)] == 0:
                            #    q2[(j,i,m,l)] = p * 1.0 / align_e[(j,m,l)]
                                #q2[j][i][m][l] = p * 1.0 / align_e[(j,m,l)]


        final_q = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:defaultdict(lambda:0))))


        for i, j, m, l in q1.keys():
            final_q[i][j][l][m] = q1[(i,j,m,l)]


        return (t1,final_q)


def main(aligned_sents):

    ba = BerkeleyAligner(aligned_sents, 10)
    A.save_model_output(aligned_sents, ba, "ba.txt")
    avg_aer = A.compute_avg_aer(aligned_sents, ba, 50)

    print ('Berkeley Aligner')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
