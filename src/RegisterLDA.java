package edu.umass.cs.wallach.cluster;

import java.io.*;
import java.util.*;

import gnu.trove.*;

import cc.mallet.types.*;
import cc.mallet.util.*;

public class RegisterLDA {

  // observed counts

  private RegisterScore registerScore;
  private SwitchScore switchScore;
  private TopicWordScore topicWordScore;
  private RegisterWordScore registerWordScore;
  private DocTopicScore docTopicScore;

  private int W, T, D, R; // constants

  private int C = 1;

  private int[][] x, z; // switch and topic assignments

  private LogRandoms rng; // random number generator

  // computes P(w, r, x, z) using the predictive distribution

  private double logProb(Corpus docs) {

    double logProb = 0;

    registerScore.resetCounts();
    switchScore.resetCounts();
    topicWordScore.resetCounts();
    registerWordScore.resetCounts();
    docTopicScore.resetCounts();

    for (int d=0; d<D; d++) {

      int r = docs.getDocument(d).getRegister();

      logProb += Math.log(registerScore.getScore(r));

      registerScore.incrementCounts(r);

      int[] fs = docs.getDocument(d).getTokens();

      int nd = fs.length;

      for (int i=0; i<nd; i++) {

        int w = fs[i];
        int k = x[d][i];
        int j = z[d][i];

        logProb += Math.log(switchScore.getScore(k, 0));

        switchScore.incrementCounts(k, 0);

        if (k == 0) {

          assert j != -1;

          logProb += Math.log(topicWordScore.getScore(w, j) * docTopicScore.getScore(j, d));

          topicWordScore.incrementCounts(w, j);
          docTopicScore.incrementCounts(j, d);
        }
        else {

          assert j == -1;

          logProb += Math.log(registerWordScore.getScore(w, r));

          registerWordScore.incrementCounts(w, r);

        }
      }
    }

    return logProb;
  }

  private void sampleVariables(Corpus docs, boolean init) {

    // resample everything

    for (int d=0; d<D; d++) {

      int rNew = -1;

      int[] fs = docs.getDocument(d).getTokens();

      int nd = fs.length;

      if (init) {

        if (R > 1)
          rNew = rng.nextInt(R);
        else
          rNew = 0;

        docs.getDocument(d).setRegister(rNew);

        registerScore.incrementCounts(rNew);
      }
      else {

        int rOld = docs.getDocument(d).getRegister();

        registerScore.decrementCounts(rOld);

        for (int i=0; i<nd; i++)
          if (x[d][i] == 1)
            registerWordScore.decrementCounts(fs[i], rOld);

        if (R > 1) {

          double[] logDist = new double[R];

          for (int r=0; r<R; r++) {

            double logScore = Math.log(registerScore.getScore(r));

            for (int i=0; i<nd; i++)
              if (x[d][i] == 1) {

                int w = fs[i];

                logScore += Math.log(registerWordScore.getScore(w, r));

                registerWordScore.incrementCounts(w, r);
              }

            logDist[r] = logScore;

            for (int i=0; i<nd; i++)
              if (x[d][i] == 1)
                registerWordScore.decrementCounts(fs[i], r);
          }

          rNew = rng.nextDiscreteLogDist(logDist);
        }
        else
          rNew = 0;

        docs.getDocument(d).setRegister(rNew);

        registerScore.incrementCounts(rNew);

        for (int i=0; i<nd; i++)
          if (x[d][i] == 1)
            registerWordScore.incrementCounts(fs[i], rNew);
      }

      if (init) {
        x[d] = new int[nd];
        z[d] = new int[nd];
      }

      for (int i=0; i<nd; i++) {

        int w = fs[i];
        int kOld = x[d][i];
        int jOld = z[d][i];

        if (!init) {

          switchScore.decrementCounts(kOld, 0);

          if (kOld == 0) {
            topicWordScore.decrementCounts(w, jOld);
            docTopicScore.decrementCounts(jOld, d);
          }
          else
            registerWordScore.decrementCounts(w, rNew);
        }

        double[] dist = new double[T+1];
        double distSum = 0.0;

        for (int j=0; j<T; j++) {

          double score = switchScore.getScore(0, 0) * topicWordScore.getScore(w, j) * docTopicScore.getScore(j, d);

          dist[j] = score;
          distSum += score;
        }

        double score = switchScore.getScore(1, 0) * registerWordScore.getScore(w, rNew);

        dist[T] = score;
        distSum += score;

        int jNew = rng.nextDiscrete(dist, distSum);

        if (jNew < T) {

          x[d][i] = 0;
          z[d][i] = jNew;

          switchScore.incrementCounts(0, 0);
          topicWordScore.incrementCounts(w, jNew);
          docTopicScore.incrementCounts(jNew, d);
        }
        else {

          x[d][i] = 1;
          z[d][i] = -1;

          switchScore.incrementCounts(1, 0);
          registerWordScore.incrementCounts(w, rNew);
        }
      }
    }
  }

  // estimate topics

  public void estimate(Corpus docs, TIntIntHashMap unseenCounts, int[][] xInit, int[][] zInit, int itnOffset, int T, int R, double[] alpha, double[] gamma, double[] beta, double[] delta, double[] sigma, int numItns, int printInterval, int saveStateInterval, boolean[] sample, String documentTopicsFileName, String topicWordsFileName, String registerWordsFileName, String topicSummaryFileName, String registerSummaryFileName, String stateFileName, String alphaFileName, String gammaFileName, String betaFileName, String deltaFileName, String sigmaFileName, String logProbFileName) {

    boolean append = false;

    if ((xInit == null) || (zInit == null))
      assert itnOffset == 0;
    else {
      assert itnOffset >= 0;
      append = true;
    }

    Alphabet wordDict = docs.getWordDict();

    assert (saveStateInterval == 0) || (numItns % saveStateInterval == 0);

    rng = new LogRandoms(1000);

    this.T = T;
    this.R = R;

    W = wordDict.size();
    D = docs.size();

    System.out.println("Num docs: " + D);
    System.out.println("Num words in vocab: " + W);
    System.out.println("Num topics: " + T);
    System.out.println("Num registers: " + R);

    registerScore = new RegisterScore(R, sigma);
    switchScore = new SwitchScore(C, gamma, new double[] { 0.5, 0.5 });
    topicWordScore = new TopicWordScore(W, T, beta, unseenCounts);
    registerWordScore = new RegisterWordScore(W, R, delta, unseenCounts);
    docTopicScore = new DocTopicScore(T, D, alpha, "minimal");

    if ((xInit == null) || (zInit == null)) {

      x = new int[D][];
      z = new int[D][];

      sampleVariables(docs, true); // initialize switch and topic assignments
    }
    else {

      x = xInit;
      z = zInit;

      for (int d=0; d<D; d++) {

        int r = docs.getDocument(d).getRegister();

        assert r != -1;

        int[] fs = docs.getDocument(d).getTokens();

        int nd = fs.length;

        for (int i=0; i<nd; i++) {

          int w = fs[i];
          int k = x[d][i];
          int j = z[d][i];

          switchScore.incrementCounts(k, 0);

          if (k == 0) {
            topicWordScore.incrementCounts(w, j);
            docTopicScore.incrementCounts(j, d);
          }
          else
            registerWordScore.incrementCounts(w, r);
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

        sampleVariables(docs, false);

        if (sample[0])
          docTopicScore.sampleAlpha(docs, x, z, rng, 5, 1.0);

        if (sample[1])
          switchScore.sampleGamma(docs, x, rng, 5, 1.0);

        if (sample[2])
          topicWordScore.sampleBeta(docs, x, z, rng, 5, 1.0);

        if (sample[3])
          registerWordScore.sampleDelta(docs, x, rng, 5, 1.0);

        if (sample[4])
          registerScore.sampleSigma(docs, rng, 5, 1.0);

        if (printInterval != 0) {
          if (s % printInterval == 0) {
            System.out.println();
            topicWordScore.print(wordDict, 0.0, 10, true, null);
            registerWordScore.print(wordDict, 0.0, 10, true, null);

            logProbWriter.println(logProb(docs));
            logProbWriter.flush();
          }
        }

        if ((saveStateInterval != 0) && (s % saveStateInterval == 0)) {
          if (stateFileName != null)
            docs.printFeatures(z, stateFileName + "." + (itnOffset + s));
          if (alphaFileName != null)
            docTopicScore.printAlpha(alphaFileName + "." + (itnOffset + s));
          if (gammaFileName != null)
            switchScore.printGamma(gammaFileName + "." + (itnOffset + s));
          if (betaFileName != null)
            topicWordScore.printBeta(betaFileName + "." + (itnOffset + s));
          if (deltaFileName != null)
            registerWordScore.printDelta(deltaFileName + "." + (itnOffset + s));
          if (sigmaFileName != null)
            registerScore.printSigma(sigmaFileName + "." + (itnOffset + s));
        }
      }

      Timer.printTimingInfo(start, System.currentTimeMillis());

      if (saveStateInterval == 0) {
        if (stateFileName != null)
          docs.printFeatures(z, stateFileName);
        if (alphaFileName != null)
          docTopicScore.printAlpha(alphaFileName);
        if (gammaFileName != null)
          switchScore.printGamma(gammaFileName);
        if (betaFileName != null)
          topicWordScore.printBeta(betaFileName);
        if (deltaFileName != null)
          registerWordScore.printDelta(deltaFileName);
        if (sigmaFileName != null)
          registerScore.printSigma(sigmaFileName);
      }

      if (documentTopicsFileName != null)
        docTopicScore.print(docs, documentTopicsFileName);
      if (topicWordsFileName != null)
        topicWordScore.print(wordDict, topicWordsFileName);
      if (registerWordsFileName != null)
        registerWordScore.print(wordDict, registerWordsFileName);

      if (topicSummaryFileName != null)
        topicWordScore.print(wordDict, 0.0, 10, true, topicSummaryFileName);
      if (registerSummaryFileName != null)
        registerWordScore.print(wordDict, 0.0, 10, true, registerSummaryFileName);

      logProbWriter.close();
    }
    catch (IOException e) {
      System.out.println(e);
    }
  }

  public double[] getAlpha() {

    return docTopicScore.getAlpha();
  }

  public double[] getGamma() {

    return switchScore.getGamma();
  }

  public double[] getBeta() {

    return topicWordScore.getBeta();
  }

  public double[] getDelta() {

    return registerWordScore.getDelta();
  }

  public double[] getSigma() {

    return registerScore.getSigma();
  }

  public int[][] getSwitch() {

    return x;
  }

  public int[][] getTopics() {

    return z;
  }
}
