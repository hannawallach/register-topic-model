package edu.umass.cs.wallach.cluster;

import java.io.*;
import java.util.*;
import java.util.zip.*;

import gnu.trove.*;

import cc.mallet.types.*;
import cc.mallet.util.Maths;

public class RegisterWordScore {

  // observed counts

  private int[][] wordRegisterCounts; // N_{w|r}
  private int[] wordRegisterCountsNorm; // N_{.|r}

  private int[][] wordRegisterCountsTrain;
  private int[] wordRegisterCountsNormTrain;

  private TIntIntHashMap unseenCounts;

  private int W, R; // constants

  // hyperparameters

  private double[] delta;

  private boolean resetToTrain = false;

  // create a score function with zero counts

  public RegisterWordScore(int W, int R, double[] delta, TIntIntHashMap unseenCounts) {

    this.W = W;
    this.R = R;

    assert delta.length == 1;
    this.delta = delta;

    this.unseenCounts = unseenCounts;

    // allocate space for counts

    wordRegisterCounts = new int[W][R];
    wordRegisterCountsNorm = new int[R];
  }

  public double getScore(int w, int r) {

    double score = 1.0 / W;

    int nwr = wordRegisterCounts[w][r];
    int nr = wordRegisterCountsNorm[r];

    score *= delta[0] / (nr + delta[0]);
    score += nwr / (nr + delta[0]);

    if (unseenCounts != null)
      if (unseenCounts.containsKey(w))
        score /= (double) unseenCounts.get(w);

    return score;
  }

  public double getScoreNoPrior(int w, int r) {

    int nr = wordRegisterCountsNorm[r];

    if (nr == 0)
      return 0.0;
    else {

      double score = (double) wordRegisterCounts[w][r] / (double) nr;

      if (unseenCounts != null)
        if (unseenCounts.containsKey(w))
          score /= (double) unseenCounts.get(w);

      return score;
    }
  }

  public void incrementCounts(int w, int r) {

    wordRegisterCounts[w][r]++;
    wordRegisterCountsNorm[r]++;
  }

  public void decrementCounts(int w, int r) {

    wordRegisterCounts[w][r]--;
    wordRegisterCountsNorm[r]--;
  }

  // this must be called before processing test data

  public void lock() {

    wordRegisterCountsTrain = new int[W][];

    for (int w=0; w<W; w++)
      wordRegisterCountsTrain[w] = wordRegisterCounts[w].clone();

    wordRegisterCountsNormTrain = wordRegisterCountsNorm.clone();

    resetToTrain = true;
  }

  public void resetCounts() {

    if (resetToTrain) {

      for (int w=0; w<W; w++)
        wordRegisterCounts[w] = wordRegisterCountsTrain[w].clone();

      wordRegisterCountsNorm = wordRegisterCountsNormTrain.clone();
    }
    else {

      for (int w=0; w<W; w++)
        Arrays.fill(wordRegisterCounts[w], 0);

      Arrays.fill(wordRegisterCountsNorm, 0);
    }
  }

  // computes log prob using the predictive distribution

  public double logProb(Corpus docs, int[][] x) {

    double logProb = 0.0;

    resetCounts();

    int D = docs.size();

    for (int d=0; d<D; d++) {

      int r = docs.getDocument(d).getRegister();

      int[] fs = docs.getDocument(d).getTokens();

      int nd = fs.length;

      for (int i=0; i<nd; i++) {

        int w = fs[i];
        int k = x[d][i];

        if (k == 1) {

          logProb += Math.log(getScore(w, r));

          incrementCounts(w, r);
        }
      }
    }

    return logProb;
  }

  public double[] getDelta() {

    return delta;
  }

  public void printDelta(String fileName) {

    try {

      PrintWriter pw = new PrintWriter(fileName);

      for (int i=0; i<delta.length; i++)
        pw.println(delta[i]);

      pw.close();
    }
    catch (IOException e) {
      System.out.println(e);
    }
  }

  private double logProb(Corpus docs, int[][] x, double[] newLogDelta) {

    double[] oldDelta = delta.clone();

    for (int i=0; i<delta.length; i++)
      delta[i] = Math.exp(newLogDelta[i]);

    double logProb = logProb(docs, x);

    for (int i=0; i<delta.length; i++)
      delta[i] = oldDelta[i];

    return logProb;
  }

  public void sampleDelta(Corpus docs, int[][] x, LogRandoms rng, int numIterations, double stepSize) {

    int I = delta.length;

    double[] rawParam = new double[I];
    double rawParamSum = 0.0;

    for (int i=0; i<I; i++) {
      rawParam[i] = Math.log(delta[i]);
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
      delta[i] = Math.exp(rawParam[i]);
  }

  public void print(Alphabet dict, String fileName) {

    print(dict, 0.0, -1, false, fileName);
  }

  public void print(Alphabet dict, double threshold, int numWords, boolean summary, String fileName) {

    assert dict.size() == W;

    try {

      PrintStream pw = null;

      if (fileName == null)
        pw = new PrintStream(System.out, true);
      else {

        pw = new PrintStream(new GZIPOutputStream(new BufferedOutputStream(new FileOutputStream(new File(fileName)))));

        if (!summary)
          pw.println("#register typeindex type proportion");
      }

      Probability[] probs = new Probability[W];

      for (int r=0; r<R; r++) {
        for (int w=0; w<W; w++)
          probs[w] = new Probability(w, getScore(w, r));

        Arrays.sort(probs);

        if ((numWords > W) || (numWords < 0))
          numWords = W;

        StringBuffer line = new StringBuffer();

        for (int i=0; i<numWords; i++) {

          // break if there are no more words whose proportion is
          // greater than zero or threshold...

          if ((probs[i].prob == 0) || (probs[i].prob < threshold))
            break;

          if ((fileName == null) || summary){
            line.append(dict.lookupObject(probs[i].index));
            line.append(" ");
          }
          else {
            pw.print(r); pw.print(" ");
            pw.print(probs[i].index); pw.print(" ");
            pw.print(dict.lookupObject(probs[i].index)); pw.print(" ");
            pw.print(probs[i].prob); pw.println();
          }
        }

        String string = line.toString();

        if ((fileName == null) || summary)
          if (!string.equals(""))
            pw.println("Register " + r + ": " + string);
      }

      if (fileName != null)
        pw.close();
    }
    catch (IOException e) {
      System.out.println(e);
    }
  }
}
