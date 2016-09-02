# import getopt
from argparse import ArgumentParser

from corpusLoader import *
from topicvecDir import topicvecDir
from utils import *
import mkl

mkl.set_num_threads(3)

config = dict(
    unigramFilename="/data/wikipedia/2016-06-21/topic-models/topic.20news.50-1500.with-classes/model.dim-50.skip-gram.embedding.restricted.vocab.counts",
    word_vec_file="/data/wikipedia/2016-06-21/topic-models/topic.20news.50-1500.with-classes/dim-50.skip-gram.embedding.model.restricted.vocab.embedding",
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
    Mstep_sample_topwords=25000,
    # normalize by the sum of Em when updating topic embeddings
    # to avoid too big gradients
    grad_scale_Em_base=10000,
    topW=10,
    # when topTopicMassFracPrintThres = 0, print all topics
    topTopicMassFracPrintThres=0,
    alpha0=0.1,
    alpha1=0.1,
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


def usage():
    print """Usage: topicExp.py -s                corpus_name set_name(s)
                   -p topic_vec_file corpus_name set_name(s)
                   [ -w ]            corpus_name set_name(s)
  set_name(s): 'train', 'test' or 'train,test' (will save in separate files)
  -s:          Train on separate categories
  -w:          Dump words only (no inference of topics)"""


corpus2loader = {'20news': load_20news, 'reuters': load_reuters}


def main():
    start_time = time.time()
    onlyDumpWords = False
    separateCatTraining = False

    parser = ArgumentParser("topicvec")
    parser.add_argument("--corpus", type=str)
    parser.add_argument("--vocabulary", type=str)
    parser.add_argument("--embeddings", type=str)
    parser.add_argument("--set-names", type=str)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--results-folder", type=str)
    args = parser.parse_args()

    try:
        os.mkdir(args.results_folder)
    except OSError:
        pass

    corpusName = args.corpus
    setNames = args.set_names.split(",")
    MAX_ITERS = args.max_iterations
    config["unigramFilename"] = args.vocabulary
    config["word_vec_file"] = args.embeddings

    if MAX_ITERS > 0:
        config['MAX_EM_ITERS'] = MAX_ITERS

    loader = corpus2loader[corpusName]

    for si, setName in enumerate(setNames):
        print "Process set '%s':" % setName

        _, orig_docs_words, orig_docs_name, orig_docs_cat, cats_docsWords, \
        cats_docNames, category_names = loader(setName)
        catNum = len(category_names)
        basename = args.results_folder
        config['logfilename'] = args.results_folder + "/log"

        # write_original_docs(basename, orig_docs_words, setDocNum)

        if si == 0:
            topicvec = topicvecDir(**config)
            out = topicvec.genOutputter(0)

        docs_idx = topicvec.setDocs(orig_docs_words, orig_docs_name)
        docs_name = [orig_docs_name[i] for i in docs_idx]
        docs_cat = [orig_docs_cat[i] for i in docs_idx]
        readDocNum = len(docs_idx)
        out("%d docs left after filtering empty docs" % (readDocNum))
        assert readDocNum == topicvec.D, "Returned %d doc idx != %d docs in Topicvec" % (readDocNum, topicvec.D)

        # write_stanford_bow_format(basename, category_names, docs_cat, docs_name, readDocNum, topicvec)
        # write_sLDA_bow_format(basename, docs_cat, readDocNum, topicvec, wid2compactId)
        # write_svm_bow_format(basename, docs_cat, readDocNum, topicvec, wid2compactId)

        if onlyDumpWords:
            continue

        # load topics from a file, infer the topic proportions, and save the proportions
        best_last_Ts, Em, docs_Em, Pi = topicvec.inference()
        # Em.shape: (50,)
        # len(Pi): num_documents, Pi[0].shape: (37, 50)
        # docs_Em.shape = (num_documents, 50)

        topic_lines = topicvec.printTopWordsInTopic(topicvec.docs_theta, False)
        with open(args.results_folder + "/topics", "w") as f:
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
    main()
