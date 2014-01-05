package edu.umass.cs.wallach.cluster;

import java.io.*;
import java.util.*;
import java.util.zip.*;

import gnu.trove.*;

import cc.mallet.types.*;
import cc.mallet.util.Maths;

public class RegisterScore {

  // observed counts

  private int[] registerCounts; // N_{r} = sum_d delta(r_d = r)
  private int registerCountsNorm; // N_{.} = D

  private int[] registerCountsTrain;
  private int registerCountsNormTrain;

  private int R; // constants

  // hyperparameters

  private double[] sigma;

  private boolean resetToTrain = false;

  // create a score function with zero counts

  public RegisterScore(int R, double[] sigma) {

    this.R = R;

    assert sigma.length == 1;
    this.sigma = sigma;

    // allocate space for counts

    registerCounts = new int[R];
    registerCountsNorm = 0;
  }

  public double getScore(int r) {

    double score = 1.0 / R;

    int nr = registerCounts[r];
    int n = registerCountsNorm;

    score *= sigma[0] / (n + sigma[0]);
    score += nr / (n + sigma[0]);

    return score;
  }

  public double getScoreNoPrior(int r) {

    int n = registerCountsNorm;

    if (n == 0)
      return 0.0;
    else
      return (double) registerCounts[r] / (double) n;
  }

  public void incrementCounts(int r) {

    registerCounts[r]++;
    registerCountsNorm++;
  }

  public void decrementCounts(int r) {

    registerCounts[r]--;
    registerCountsNorm--;
  }

  // this must be called before processing test data

  public void lock(int numTestDocs) {

    registerCountsTrain = registerCounts.clone();
    registerCountsNormTrain = registerCountsNorm;

    resetToTrain = true;
  }

  public void resetCounts() {

    if (resetToTrain) {
      registerCounts = registerCountsTrain.clone();
      registerCountsNorm = registerCountsNormTrain;
    }
    else {
      Arrays.fill(registerCounts, 0);
      registerCountsNorm = 0;
    }
  }

  // computes log prob using the predictive distribution

  public double logProb(Corpus docs) {

    double logProb = 0.0;

    resetCounts();

    int D = docs.size();

    for (int d=0; d<D; d++) {

      int r = docs.getDocument(d).getRegister();

      logProb += Math.log(getScore(r));

      incrementCounts(r);
    }

    return logProb;
  }

  public double[] getSigma() {

    return sigma;
  }

  public void printSigma(String fileName) {

    try {

      PrintWriter pw = new PrintWriter(fileName);

      for (int i=0; i<sigma.length; i++)
        pw.println(sigma[i]);

      pw.close();
    }
    catch (IOException e) {
      System.out.println(e);
    }
  }

  private double logProb(Corpus docs, double[] newLogSigma) {

    double[] oldSigma = sigma.clone();

    for (int i=0; i<sigma.length; i++)
      sigma[i] = Math.exp(newLogSigma[i]);

    double logProb = logProb(docs);

    for (int i=0; i<sigma.length; i++)
      sigma[i] = oldSigma[i];

    return logProb;
  }

  public void sampleSigma(Corpus docs, LogRandoms rng, int numIterations, double stepSize) {

    int I = sigma.length;

    double[] rawParam = new double[I];
    double rawParamSum = 0.0;

    for (int i=0; i<I; i++) {
      rawParam[i] = Math.log(sigma[i]);
      rawParamSum += rawParam[i];
    }

    double[] l = new double[I];
    double[] r = new double[I];

    for (int s=0; s<numIterations; s++) {

      double lp = logProb(docs, rawParam) + rawParamSum;
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

        if (logProb(docs, rawParamNew) + rawParamNewSum > lpNew)
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
      sigma[i] = Math.exp(rawParam[i]);
  }
}
