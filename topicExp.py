# import getopt
from argparse import ArgumentParser

from corpusLoader import *
from itertools import groupby
from topicvecDir import topicvecDir
from utils import *
import mkl

mkl.set_num_threads(1)

config = dict(
    unigramFilename="foo",
    word_vec_file="bar",
    # word_vec_file = "25000-500-EM.vec",
    # word_vec_file = "7929-400-EM.vec",
    K=50,
    # for separate category training, each category has 10 topics, totalling 200
    sepK_20news=15,
    sepK_reuters=12,
    # set it to 0 to disable the removal of very small topics
    topTopicMassFracThres=0.05,
    N0=500,
    max_l=7,
    init_l=1,
    # cap the norm of the gradient of topics to avoid too big gradients
    max_grad_norm=5,
    Mstep_sample_topwords=13740,
    # normalize by the sum of Em when updating topic embeddings
    # to avoid too big gradients
    grad_scale_Em_base=10000,
    topW=10,
    # when topTopicMassFracPrintThres = 0, print all topics
    topTopicMassFracPrintThres=0,
    alpha0=0.02,
    alpha1=0.02,
    delta=0.1,
    max_theta_to_avg_ratio=-1,
    big_theta_step_ratio=2,
    MAX_EM_ITERS=100,
    topicDiff_tolerance=2e-3,
    zero_topic0=True,
    smoothing_context_size=0,
    remove_stop=True,
    useDrdtApprox=False,
    verbose=0,
    seed=0,
    printTopic_iterNum=10,
    calcSum_pi_v_iterNum=1,
    VStep_iterNum=5
    )


def read_nips_corpus(args):
    with open(args.corpus) as f:
        lines = [l.rstrip() for l in f.readlines()]
        lines = [l.split("\t") for l in lines]
        lines = [(l[0], l[1]) for l in lines]
        lines = [(l[0], l[1].split(" ")) for l in lines]

        docs = [map(lambda x: x[1], list(sentences)) for doc, sentences in groupby(lines, lambda x: x[0])]

    return docs, map(lambda x: str(x), list(range(len(docs)))), [0] * len(docs)


def read_corpus(args):
    with open(args.vocabulary, "r") as f:
        vocab = [l.rstrip() for l in f.readlines()]
        vocab = [l.split("\t")[0] for l in vocab]

    corpus = []
    current_doc = []
    line_nr = 0

    current_name = ""
    current_class = ""
    with open(args.corpus, "r") as f:
        for line in f:
            line = line.rstrip()
            if line == "##":
                if len(current_doc) > 0:
                    corpus.append((current_doc, current_name, current_class))
                    current_doc = []
                else:
                    print "empty document at line " + line_nr
            elif "\t" in line:
                split = line.split("\t")
                current_name = split[0]
                current_class = int(split[1])
            else:
                word_id = int(line[:6])
                # topic_id = int(line[7:])
                current_doc.append(word_id)
                # topics.add(topicId)
            line_nr += 1

    corpus = [([vocab[word_id] for word_id in doc], name, clazz) for (doc, name, clazz) in corpus]

    document_sentences = []
    names = []
    classes = []
    for name, group in groupby(corpus, key=lambda x: x[1]):
        group = list(group)
        sentences = [g[0] for g in group]
        document_sentences.append(sentences)
        names.append(name)
        classes.append(group[0][2])
    return document_sentences, names, classes


def main(args):
    start_time = time.time()

    try:
        os.mkdir(args.results_folder)
    except OSError:
        pass

    corpus2loader = {'20news': load_20news, 'reuters': load_reuters}

    MAX_ITERS = args.max_iterations
    config["unigramFilename"] = args.vocabulary
    config["word_vec_file"] = args.embeddings
    config["K"] = args.num_topics + 1
    config["alpha0"] = args.alpha
    config["alpha1"] = args.alpha

    if MAX_ITERS > 0:
        config['MAX_EM_ITERS'] = MAX_ITERS

    config['logfilename'] = args.results_folder + "/log"
    topicvec = topicvecDir(**config)
    out = topicvec.genOutputter(0)

    if "/nips/" in args.corpus:
        orig_docs_words, orig_docs_name, orig_docs_cat = read_nips_corpus(args)
    else:
        orig_docs_words, orig_docs_name, orig_docs_cat = read_corpus(args)

    # _, orig_docs_words, orig_docs_name, orig_docs_cat, _, _, _ = corpus2loader["20news"]("train")

    # orig_docs_cat = their_orig_docs_cat
    # orig_docs_words = their_orig_docs_words
    # orig_docs_name = their_orig_docs_name

    # our = zip(orig_docs_name, orig_docs_words)
    # our = sorted(our, key=lambda x: x[0])
    # theirs = zip(foo_orig_docs_name, foo_orig_docs_words)
    # theirs = sorted(theirs, key=lambda x: x[0])
    basename = args.results_folder

    # write_original_docs(basename, orig_docs_words, setDocNum)

    docs_idx = topicvec.setDocs(orig_docs_words, orig_docs_name)
    docs_name = [orig_docs_name[i] for i in docs_idx]
    docs_cat = [orig_docs_cat[i] for i in docs_idx]
    readDocNum = len(docs_idx)
    out("%d docs left after filtering empty docs" % (readDocNum))
    assert readDocNum == topicvec.D, "Returned %d doc idx != %d docs in Topicvec" % (readDocNum, topicvec.D)

    # write_stanford_bow_format(basename, category_names, docs_cat, docs_name, readDocNum, topicvec)
    # write_sLDA_bow_format(basename, docs_cat, readDocNum, topicvec, wid2compactId)
    # write_svm_bow_format(basename, docs_cat, readDocNum, topicvec, wid2compactId)

    # load topics from a file, infer the topic proportions, and save the proportions
    best_last_Ts, Em, docs_Em, Pi = topicvec.inference(args.results_folder)
    # Em.shape: (50,)
    # len(Pi): num_documents, Pi[0].shape: (37, 50)
    # docs_Em.shape = (num_documents, 50)

    topic_lines = topicvec.printTopWordsInTopic(topicvec.docs_theta, False)
    with open(args.results_folder + "/iteration-" + str(args.max_iterations) + ".topics", "w") as f:
        f.writelines([l + "\n" for l in topic_lines])

    topicvec.topW = 500
    topic_lines = topicvec.printTopWordsInTopic(topicvec.docs_theta, False)
    with open(args.results_folder + "/iteration-" + str(args.max_iterations) + ".500.topics", "w") as f:
        f.writelines([l + "\n" for l in topic_lines])

    best_it, best_T, best_loglike = best_last_Ts[0]
    # last_it, last_T, last_loglike = best_last_Ts[1]

    save_matrix_as_text(basename + "/topics_matrix", "best topics", best_T)
    # save_matrix_as_text(doc_name + "-em%d-last.topic.vec" % last_it, "last topics", last_T)

    save_matrix_as_text(basename + "/document-topics", "topic proportion", docs_Em,
                        docs_cat, docs_name, colSep="\t")

    end_time = time.time()
    duration = int(end_time - start_time)

    with open(args.results_folder + "/runtime.txt", "w") as runtime_file:
        runtime_file.write(str(duration) + "\n")


def write_word_mapping(basename, compactIds_word, sorted_wids, uniq_wid_num):
    print "Word mapping created: %d -> %d" % (sorted_wids[-1], uniq_wid_num)
    id2word_filename = "%s.id2word.txt" % basename
    ID2WORD = open(id2word_filename, "w")
    for i in xrange(uniq_wid_num):
        ID2WORD.write("%d\t%s\n" % (i, compactIds_word[i]))
    ID2WORD.close()


def write_svm_bow_format(basename, docs_cat, readDocNum, topicvec, wid2compactId):
    # dump words in libsvm/svmlight format
    svmbow_filename = "%s.svm-bow.txt" % basename
    SVMBOW = open(svmbow_filename, "w")
    for i in xrange(readDocNum):
        wids = topicvec.docs_wids[i]
        cwid2freq = {}
        for wid in wids:
            # cwid could be 0, but svm feature index cannot be 0
            # +1 to avoid 0 being used as a feature index
            cwid = wid2compactId[wid] + 1
            if cwid in cwid2freq:
                cwid2freq[cwid] += 1
            else:
                cwid2freq[cwid] = 1
        catID = docs_cat[i]
        sorted_cwids = sorted(cwid2freq.keys())
        SVMBOW.write("%d" % (catID + 1))
        for cwid in sorted_cwids:
            SVMBOW.write(" %d:%d" % (cwid, cwid2freq[cwid]))
        SVMBOW.write("\n")
    SVMBOW.close()
    print "%d docs saved in '%s' in svm bow format" % (readDocNum, svmbow_filename)


def write_sLDA_bow_format(basename, docs_cat, readDocNum, topicvec, wid2compactId):
    # dump words in sLDA format
    slda_bow_filename = "%s.slda-bow.txt" % basename
    slda_label_filename = "%s.slda-label.txt" % basename
    SLDA_BOW = open(slda_bow_filename, "w")
    SLDA_LABEL = open(slda_label_filename, "w")
    for i in xrange(readDocNum):
        wids = topicvec.docs_wids[i]
        # compact wid to freq
        cwid2freq = {}
        for wid in wids:
            cwid = wid2compactId[wid]
            if cwid in cwid2freq:
                # wid could be 0, but svm feature index cannot be 0
                # +1 to avoid 0 being used as a feature index
                cwid2freq[cwid] += 1
            else:
                cwid2freq[cwid] = 1
        catID = docs_cat[i]
        sorted_cwids = sorted(cwid2freq.keys())
        uniq_wid_num = len(sorted_cwids)
        # sLDA requires class lables to start from 0
        SLDA_LABEL.write("%d\n" % catID)
        SLDA_BOW.write("%d" % uniq_wid_num)
        for cwid in sorted_cwids:
            SLDA_BOW.write(" %d:%d" % (cwid, cwid2freq[cwid]))
        SLDA_BOW.write("\n")
    SLDA_BOW.close()
    SLDA_LABEL.close()
    print "%d docs saved in '%s' and '%s' in sLDA bow format" % (readDocNum,
                                                                 slda_bow_filename, slda_label_filename)


def write_stanford_bow_format(basename, category_names, docs_cat, docs_name, readDocNum, topicvec):
    # dump words in stanford classifier format
    stanford_filename = "%s.stanford-bow.txt" % basename
    STANFORD = open(stanford_filename, "w")
    for i in xrange(readDocNum):
        wids = topicvec.docs_wids[i]
        words = [topicvec.vocab[j] for j in wids]
        text = " ".join(words)
        catID = docs_cat[i]
        category = category_names[catID]
        doc_name = docs_name[i]
        STANFORD.write("%s\t%s\t%s\n" % (category, doc_name, text))
    STANFORD.close()
    print "%d docs saved in '%s' in stanford bow format" % (readDocNum, stanford_filename)


def write_original_docs(basename, orig_docs_words, setDocNum):
    # dump original words (without filtering)
    orig_filename = "%s.orig.txt" % basename
    ORIG = open(orig_filename, "w")
    for wordsInSentences in orig_docs_words:
        for sentence in wordsInSentences:
            for w in sentence:
                w = w.lower()
                ORIG.write("%s " % w)
        ORIG.write("\n")
    ORIG.close()
    print "%d original docs saved in '%s'" % (setDocNum, orig_filename)


if __name__ == "__main__":
    parser = ArgumentParser("topicvec")
    parser.add_argument("--corpus", type=str)
    parser.add_argument("--vocabulary", type=str)
    parser.add_argument("--embeddings", type=str)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--num-topics", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--results-folder", type=str)
    args = parser.parse_args()

    orig_corpus = args.corpus
    orig_vocabulary = args.vocabulary
    orig_embeddings = args.embeddings

    from datetime import datetime
    print datetime.now()
    sys.stdout.flush()
    main(args)
    print datetime.now()
    sys.stdout.flush()

    # for dim in [50]:
    #     for iterations in [500]:
    #         args.embeddings = args
    #         args.max_iterations = iterations
    #         args.vocabulary = orig_vocabulary.replace("dim-XXX", "dim-%d" % dim)
    #         args.embeddings = orig_embeddings.replace("dim-XXX", "dim-%d" % dim)
    #         args.corpus = orig_corpus.replace("dim-XXX", "dim-%d" % dim)
    #         args.results_folder = "results/corpus-orig.dim-%d.iterations-%d" % (dim, iterations)
    #
    #         base_corpus = os.path.basename(args.corpus)
    #         base_vocab = os.path.basename(args.vocabulary)
    #         base_embeddings = os.path.basename(args.embeddings)
    #         print "%s - %s - %s" % (base_corpus, base_vocab, base_embeddings)
    #         sys.stdout.flush()
    #         main(args)
