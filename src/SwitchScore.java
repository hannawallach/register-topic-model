package edu.umass.cs.wallach.cluster;

import java.io.*;
import java.util.*;
import java.util.zip.*;

import gnu.trove.*;

import cc.mallet.types.*;
import cc.mallet.util.Maths;

public class SwitchScore {

  // observed counts

  private int[][] switchCounts; // N_{k|c}
  private int[] switchCountsNorm; // N_{.|c}

  private int[][] switchCountsTrain;
  private int[] switchCountsNormTrain;

  private int C; // constants

  // hyperparameters

  private double[] gamma;
  private double[] triangle;

  private boolean resetToTrain = false;

  // create a score function with zero counts

  public SwitchScore(int C, double[] gamma, double[] triangle) {

    this.C = C;

    assert gamma.length == 1;
    this.gamma = gamma;

    assert triangle.length == 2;
    this.triangle = triangle;

    // allocate space for counts

    switchCounts = new int[2][C];
    switchCountsNorm = new int[C];
  }

  public double getScore(int k, int c) {

    double score = triangle[k];

    int nkc = switchCounts[k][c];
    int nc = switchCountsNorm[c];

    score *= gamma[0] / (nc + gamma[0]);
    score += nkc / (nc + gamma[0]);

    return score;
  }

  public double getScoreNoPrior(int k, int c) {

    int nc = switchCountsNorm[c];

    if (nc == 0)
      return 0.0;
    else
      return (double) switchCounts[k][c] / (double) nc;
  }

  public void incrementCounts(int k, int c) {

    switchCounts[k][c]++;
    switchCountsNorm[c]++;
  }

  public void decrementCounts(int k, int c) {

    switchCounts[k][c]--;
    switchCountsNorm[c]--;
  }

  // this must be called before processing test data

  public void lock(int numTestDocs) {

    switchCountsTrain = new int[2][];

    for (int k=0; k<2; k++)
      switchCountsTrain[k] = switchCounts[k].clone();

    switchCountsNormTrain = switchCountsNorm.clone();

    resetToTrain = true;
  }

  public void resetCounts() {

    if (resetToTrain) {

      for (int k=0; k<2; k++)
        switchCounts[k] = switchCountsTrain[k].clone();

      switchCountsNorm = switchCountsNormTrain.clone();
    }
    else {

      for (int k=0; k<2; k++)
        Arrays.fill(switchCounts[k], 0);

      Arrays.fill(switchCountsNorm, 0);
    }
  }

  // computes log prob using the predictive distribution

  public double logProb(Corpus docs, int[][] x) {

    double logProb = 0.0;

    resetCounts();

    int D = docs.size();

    assert x.length == D;

    for (int d=0; d<D; d++) {

      int[] fs = docs.getDocument(d).getTokens();

      int nd = fs.length;

      int[] cs = docs.getDocument(d).getChunks();

      if (cs != null)
        assert cs.length == nd;

      for (int i=0; i<nd; i++) {

        int k = x[d][i];
        int c = (cs != null) ? cs[i] : 0;

        logProb += Math.log(getScore(k, c));

        incrementCounts(k, c);
      }
    }

    return logProb;
  }

  public double[] getGamma() {

    return gamma;
  }

  public void printGamma(String fileName) {

    try {

      PrintWriter pw = new PrintWriter(fileName);

      for (int i=0; i<gamma.length; i++)
        pw.println(gamma[i]);

      pw.close();
    }
    catch (IOException e) {
      System.out.println(e);
    }
  }

  private double logProb(Corpus docs, int[][] x, double[] newLogGamma) {

    double[] oldGamma = gamma.clone();

    for (int i=0; i<gamma.length; i++)
      gamma[i] = Math.exp(newLogGamma[i]);

    double logProb = logProb(docs, x);

    for (int i=0; i<gamma.length; i++)
      gamma[i] = oldGamma[i];

    return logProb;
  }

  public void sampleGamma(Corpus docs, int[][] x, LogRandoms rng, int numIterations, double stepSize) {

    int I = gamma.length;

    double[] rawParam = new double[I];
    double rawParamSum = 0.0;

    for (int i=0; i<I; i++) {
      rawParam[i] = Math.log(gamma[i]);
      rawParamSum += rawParam[i];
    }

    double[] l = new double[I];
    double[] r = new double[I];

    for (int s=0; s<numIterations; s++) {

      double lp = logProb(docs, x, rawParam) + rawParamSum;
      double lpNew = Math.log(rng.nextUniform()) + lp;

      for (int i=0; i<I; i++) {
        l[i] = rawParam[i] - rng.nextUniform() * stepSize;
        r[i] = l[i] + stepSize;
      }

      double[] rawParamNew = new double[I];
      double rawParamNewSum = 0.0;

      while (true) {

        rawParamNewSum = 0.0;

        for (int i=0; i<I; i++) {
          rawParamNew[i] = l[i] + rng.nextUniform() * (r[i] - l[i]);
          rawParamNewSum += rawParamNew[i];
        }

        if (logProb(docs, x, rawParamNew) + rawParamNewSum > lpNew)
          break;
        else
          for (int i=0; i<I; i++)
            if (rawParamNew[i] < rawParam[i])
              l[i] = rawParamNew[i];
            else
              r[i] = rawParamNew[i];
      }

      rawParam = rawParamNew;
      rawParamSum = rawParamNewSum;
    }

    for (int i=0; i<I; i++)
      gamma[i] = Math.exp(rawParam[i]);
  }
}
