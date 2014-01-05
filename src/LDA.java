package edu.umass.cs.wallach.cluster;

import java.io.*;
import java.util.*;

import gnu.trove.*;

import cc.mallet.types.*;
import cc.mallet.util.*;

public class LDA {

  // observed counts

  private TopicWordScore topicWordScore;
  private DocTopicScore docTopicScore;

  private int W, T, D; // constants

  private int[][] z; // topic assignments

  private LogRandoms rng; // random number generator

  private double getScore(int w, int j, int d) {

    return topicWordScore.getScore(w, j) * docTopicScore.getScore(j, d);
  }

  // computes P(w, z) using the predictive distribution

  private double logProb(Corpus docs) {

    double logProb = 0;

    topicWordScore.resetCounts();
    docTopicScore.resetCounts();

    for (int d=0; d<D; d++) {

      int[] fs = docs.getDocument(d).getTokens();

      int nd = fs.length;

      for (int i=0; i<nd; i++) {

        int w = fs[i];
        int j = z[d][i];

        logProb += Math.log(getScore(w, j, d));

        topicWordScore.incrementCounts(w, j);
        docTopicScore.incrementCounts(j, d);
      }
    }

    return logProb;
  }

  private void sampleTopics(Corpus docs, boolean init) {

    // resample topics

    for (int d=0; d<D; d++) {

      int[] fs = docs.getDocument(d).getTokens();

      int nd = fs.length;

      if (init)
        z[d] = new int[nd];

      for (int i=0; i<nd; i++) {

        int w = fs[i];
        int oldTopic = z[d][i];

        if (!init) {
          topicWordScore.decrementCounts(w, oldTopic);
          docTopicScore.decrementCounts(oldTopic, d);
        }

        // build a distribution over topics

        double dist[] = new double[T];
        double distSum = 0.0;

        for (int j=0; j<T; j++) {

          double score = getScore(w, j, d);

          dist[j] = score;
          distSum += score;
        }

        int newTopic = rng.nextDiscrete(dist, distSum);

        z[d][i] = newTopic;

        topicWordScore.incrementCounts(w, newTopic);
        docTopicScore.incrementCounts(newTopic, d);
      }
    }
  }

  // estimate topics

  public void estimate(Corpus docs, TIntIntHashMap unseenCounts, int[][] zInit, int itnOffset, int T, double[] alpha, double[] beta, int numItns, int printInterval, int saveStateInterval, boolean[] sample, String documentTopicsFileName, String topicWordsFileName, String topicSummaryFileName, String stateFileName, String alphaFileName, String betaFileName, String logProbFileName) {

    boolean append = false;

    if (zInit == null)
      assert itnOffset == 0;
    else {
      assert itnOffset >= 0;
      append = true;
    }

    Alphabet wordDict = docs.getWordDict();

    assert (saveStateInterval == 0) || (numItns % saveStateInterval == 0);

    rng = new LogRandoms(1000);

    this.T = T;

    W = wordDict.size();
    D = docs.size();

    System.out.println("Num docs: " + D);
    System.out.println("Num words in vocab: " + W);
    System.out.println("Num topics: " + T);

    topicWordScore = new TopicWordScore(W, T, beta, unseenCounts);
    docTopicScore = new DocTopicScore(T, D, alpha, "minimal");

    if (zInit == null) {

      z = new int[D][];
      sampleTopics(docs, true); // initialize topic assignments
    }
    else {

      z = zInit;

      for (int d=0; d<D; d++) {

        int[] fs = docs.getDocument(d).getTokens();

        int nd = fs.length;

        for (int i=0; i<nd; i++) {

          int w = fs[i];
          int topic = z[d][i];

          topicWordScore.incrementCounts(w, topic);
          docTopicScore.incrementCounts(topic, d);
        }
      }
    }

    long start = System.currentTimeMillis();

    try {

      PrintWriter logProbWriter = new PrintWriter(new FileWriter(logProbFileName, append), true);

      // count matrices have been populated, every token has been
      // assigned to a single topic, so Gibbs sampling can start

      for (int s=1; s<=numItns; s++) {

        if (s % 10 == 0)
          System.out.print(s);
        else
          System.out.print(".");

        System.out.flush();

        sampleTopics(docs, false);

        if (sample[0])
          docTopicScore.sampleAlpha(docs, null, z, rng, 5, 1.0);

        if (sample[1])
          topicWordScore.sampleBeta(docs, null, z, rng, 5, 1.0);

        if (printInterval != 0) {
          if (s % printInterval == 0) {
            System.out.println();
            topicWordScore.print(wordDict, 0.0, 10, true, null);

            logProbWriter.println(logProb(docs));
            logProbWriter.flush();
          }
        }

        if ((saveStateInterval != 0) && (s % saveStateInterval == 0)) {
          if (stateFileName != null)
            docs.printFeatures(z, stateFileName + "." + (itnOffset + s));
          if (alphaFileName != null)
            docTopicScore.printAlpha(alphaFileName + "." + (itnOffset + s));
          if (betaFileName != null)
            topicWordScore.printBeta(betaFileName + "." + (itnOffset + s));
        }
      }

      Timer.printTimingInfo(start, System.currentTimeMillis());

      if (saveStateInterval == 0) {
        if (stateFileName != null)
          docs.printFeatures(z, stateFileName);
        if (alphaFileName != null)
          docTopicScore.printAlpha(alphaFileName);
        if (betaFileName != null)
          topicWordScore.printBeta(betaFileName);
      }

      if (documentTopicsFileName != null)
        docTopicScore.print(docs, documentTopicsFileName);
      if (topicWordsFileName != null)
        topicWordScore.print(wordDict, topicWordsFileName);

      if (topicSummaryFileName != null)
        topicWordScore.print(wordDict, 0.0, 10, true, topicSummaryFileName);

      logProbWriter.close();
    }
    catch (IOException e) {
      System.out.println(e);
    }
  }

  public double[] getAlpha() {

    return docTopicScore.getAlpha();
  }

  public double[] getBeta() {

    return topicWordScore.getBeta();
  }

  public int[][] getTopics() {

    return z;
  }
}
