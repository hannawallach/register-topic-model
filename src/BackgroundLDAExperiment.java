package edu.umass.cs.wallach.cluster;

import java.io.*;
import java.util.*;

import gnu.trove.*;

import cc.mallet.types.*;

public class BackgroundLDAExperiment {

  public static void main(String[] args) throws java.io.IOException {

    if (args.length != 7) {
      System.out.println("Usage: LDAExperiment <instance_list> <num_topics> <num_itns> <print_interval> <save_state_interval> <sample> <output_dir>");
      System.exit(1);
    }

    int index = 0;

    String instanceListFileName = args[index++];

    int T = Integer.parseInt(args[index++]); // # of topics

    int numIterations = Integer.parseInt(args[index++]); // # Gibbs iterations
    int printInterval = Integer.parseInt(args[index++]); // # iterations between printing out topics
    int saveStateInterval = Integer.parseInt(args[index++]);

    assert args[index].length() == 2;
    boolean[] sample = new boolean[4]; // whether to sample hyperparameters

    for (int i=0; i<4; i++)
      switch(args[index].charAt(i)) {
      case '0': sample[i] = false; break;
      case '1': sample[i] = true; break;
      default: System.exit(1);
      }

    index++;

    String outputDir = args[index++]; // output directory

    assert index == 7;

    // load data

    Alphabet wordDict = new Alphabet();

    Corpus docs = new Corpus(wordDict, null);

    InstanceListLoader.load(instanceListFileName, docs);

    int W = wordDict.size();

    for (int d=0; d<docs.size(); d++)
      docs.getDocument(d).setRegister(0);

    int R = 1;

    double[] alpha = new double[2];
    Arrays.fill(alpha, 0.1 * T);

    double[] gamma = new double[1];
    Arrays.fill(gamma, 1.0);

    double[] beta = new double[1];
    Arrays.fill(beta, 0.01 * W);

    double[] delta = new double[1];
    Arrays.fill(delta, 0.01 * W);

    // form output filenames

    String optionsFileName = outputDir + "/options.txt";

    String documentTopicsFileName = outputDir + "/doc_topics.txt.gz";
    String topicWordsFileName = outputDir + "/topic_words.txt.gz";
    String registerWordsFileName = outputDir + "/register_words.txt.gz";
    String topicSummaryFileName = outputDir + "/topic_summary.txt.gz";
    String registerSummaryFileName = outputDir + "/register_summary.txt.gz";
    String stateFileName = outputDir + "/state.txt.gz";
    String alphaFileName = outputDir + "/alpha.txt";
    String gammaFileName = outputDir + "/gamma.txt";
    String betaFileName = outputDir + "/beta.txt";
    String deltaFileName = outputDir + "/delta.txt";
    String logProbFileName = outputDir + "/log_prob.txt";

    PrintWriter pw = new PrintWriter(optionsFileName);

    pw.println("Instance list = " + instanceListFileName);

    int corpusLength = 0;

    for (int d=0; d<docs.size(); d++)
      corpusLength += docs.getDocument(d).getLength();

    pw.println("# tokens = " + corpusLength);

    pw.println("T = " + T);
    pw.println("# iterations = " + numIterations);
    pw.println("Print interval = " + printInterval);
    pw.println("Save state interval = " + saveStateInterval);
    pw.println("Sample alpha = " + sample[0]);
    pw.println("Sample gamma = " + sample[1]);
    pw.println("Sample beta = " + sample[2]);
    pw.println("Sample delta = " + sample[3]);
    pw.println("Date = " + (new Date()));

    pw.close();

    BackgroundLDA lda = new BackgroundLDA();

    lda.estimate(docs, null, null, null, 0, T, R, alpha, gamma, beta, delta, numIterations, printInterval, saveStateInterval, sample, documentTopicsFileName, topicWordsFileName, registerWordsFileName, topicSummaryFileName, registerSummaryFileName, stateFileName, alphaFileName, gammaFileName, betaFileName, deltaFileName, logProbFileName);

  }
}
