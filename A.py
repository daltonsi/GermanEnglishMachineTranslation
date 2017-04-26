import nltk
from nltk.corpus import comtrans
from nltk.align import IBMModel1
from nltk.align import IBMModel2

# TODO: Initialize IBM Model 1 and return the model.
def create_ibm1(aligned_sents):

    #http://nullege.com/codes/search/nltk.align.ibm1.IBMModel1
    ibm1 = IBMModel1(aligned_sents, 10)
    return ibm1

# TODO: Initialize IBM Model 2 and return the model.
def create_ibm2(aligned_sents):
    ibm2 = IBMModel2(aligned_sents, 10)
    return ibm2

# TODO: Compute the average AER for the first n sentences
#       in aligned_sents using model. Return the average AER.
# http://www.nltk.org/howto/align.html
def compute_avg_aer(aligned_sents, model, n):
    errors = []
    for x in range(0,n):
        alignment = model.align(aligned_sents[x])
        error = alignment.alignment_error_rate(aligned_sents[x])
        errors.append(error)
    average_aer = sum(errors)/n
    return average_aer

# TODO: Computes the alignments for the first 20 sentences in
#       aligned_sents and saves the sentences and their alignments
#       to file_name. Use the format specified in the assignment.
def save_model_output(aligned_sents, model, file_name):
    output_file = open(file_name, 'wb')
    for x in range(0,20):
        alignment = model.align(aligned_sents[x])
        source_words = alignment.words
        target_words = alignment.mots
        alignments = alignment.alignment
        output_file.write(str(source_words) + '\n' + str(target_words) + '\n' + str(alignments) + '\n' + str(aligned_sents[x].alignment_error_rate(alignment)) + '\n\n')


def main(aligned_sents):
    ibm1 = create_ibm1(aligned_sents)
    save_model_output(aligned_sents, ibm1, "ibm1.txt")
    avg_aer = compute_avg_aer(aligned_sents, ibm1, 50)

    print ('IBM Model 1')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))

    ibm2 = create_ibm2(aligned_sents)
    save_model_output(aligned_sents, ibm2, "ibm2.txt")
    avg_aer = compute_avg_aer(aligned_sents, ibm2, 50)

    print ('IBM Model 2')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
