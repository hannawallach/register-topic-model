package edu.umass.cs.wallach.cluster;

import java.io.*;
import java.util.*;
import java.util.zip.*;

import gnu.trove.*;

import cc.mallet.types.*;
import cc.mallet.util.Maths;

public class ChunkRegisterWordScore {

  // observed counts

  private int[][] wordRegisterChunkCounts; // N_{w|r,c}
  private int[] wordRegisterChunkCountsNorm; // N_{.|r,c}

  private int[][] wordRegisterCounts; // N_{w|r}
  private int[] wordRegisterCountsNorm; // N_{.|r}

  private int[][] wordRegisterChunkCountsTrain;
  private int[] wordRegisterChunkCountsNormTrain;

  private int[][] wordRegisterCountsTrain;
  private int[] wordRegisterCountsNormTrain;

  private TIntIntHashMap unseenCounts;

  private int W, R, C; // constants

  // hyperparameters

  private double[] delta;

  private boolean resetToTrain = false;

  private String score;

  // create a score function with zero counts

  public ChunkRegisterWordScore(int W, int R, int C, double[] delta, TIntIntHashMap unseenCounts, String score) {

    this.W = W;
    this.R = R;
    this.C = C;

    assert delta.length == 2;
    this.delta = delta;

    this.unseenCounts = unseenCounts;

    this.score = score;

    // allocate space for counts

    wordRegisterChunkCounts = new int[W][R * C];
    wordRegisterChunkCountsNorm = new int[R * C];

    wordRegisterCounts = new int[W][R];
    wordRegisterCountsNorm = new int[R];
  }

  public double getScore(int w, int r, int c) {

    double score = 1.0 / W;

    int index = (r * C) + c;

    int nwr = wordRegisterCounts[w][r];
    int nr = wordRegisterCountsNorm[r];

    score *= delta[0] / (nr + delta[0]);
    score += nwr / (nr + delta[0]);

    int nwrc = wordRegisterChunkCounts[w][index];
    int nrc = wordRegisterChunkCountsNorm[index];

    score *= delta[1] / (nrc + delta[1]);
    score += nwrc / (nrc + delta[1]);

    if (unseenCounts != null)
      if (unseenCounts.containsKey(w))
        score /= (double) unseenCounts.get(w);

    return score;
  }

  public double getScoreNoPrior(int w, int r, int c) {

    int index = (r * C) + c;

    int nrc = wordRegisterChunkCountsNorm[index];

    if (nrc == 0)
      return 0.0;
    else {

      double score = (double) wordRegisterChunkCounts[w][index] / (double) nrc;

      if (unseenCounts != null)
        if (unseenCounts.containsKey(w))
          score /= (double) unseenCounts.get(w);

      return score;
    }
  }

  public void incrementCounts(int w, int r, int c) {

    int index = (r * C) + c;

    int oldCount = wordRegisterChunkCounts[w][index]++;
    wordRegisterChunkCountsNorm[index]++;

    if (score.equals("minimal")) {
      if (oldCount == 0) {
        wordRegisterCounts[w][r]++;
        wordRegisterCountsNorm[r]++;
      }
    }
    else {
      wordRegisterCounts[w][r]++;
      wordRegisterCountsNorm[r]++;
    }
  }

  public void decrementCounts(int w, int r, int c) {

    int index = (r * C) + c;

    int oldCount = wordRegisterChunkCounts[w][index]--;
    wordRegisterChunkCountsNorm[index]--;

    if (score.equals("minimal")) {
      if (oldCount == 1) {
        wordRegisterCounts[w][r]--;
        wordRegisterCountsNorm[r]--;
      }
    }
    else {
      wordRegisterCounts[w][r]--;
      wordRegisterCountsNorm[r]--;
    }
  }

  // this must be called before processing test data

  public void lock() {

    wordRegisterChunkCountsTrain = new int[W][];

    for (int w=0; w<W; w++)
      wordRegisterChunkCountsTrain[w] = wordRegisterChunkCounts[w].clone();

    wordRegisterChunkCountsNormTrain = wordRegisterChunkCountsNorm.clone();

    wordRegisterCountsTrain = new int[W][];

    for (int w=0; w<W; w++)
      wordRegisterCountsTrain[w] = wordRegisterCounts[w].clone();

    wordRegisterCountsNormTrain = wordRegisterCountsNorm.clone();

    resetToTrain = true;
  }

  public void resetCounts() {

    if (resetToTrain) {

      for (int w=0; w<W; w++)
        wordRegisterChunkCounts[w] = wordRegisterChunkCountsTrain[w].clone();

      wordRegisterChunkCountsNorm = wordRegisterChunkCountsNormTrain.clone();

      for (int w=0; w<W; w++)
        wordRegisterCounts[w] = wordRegisterCountsTrain[w].clone();

      wordRegisterCountsNorm = wordRegisterCountsNormTrain.clone();
    }
    else {

      for (int w=0; w<W; w++)
        Arrays.fill(wordRegisterChunkCounts[w], 0);

      Arrays.fill(wordRegisterChunkCountsNorm, 0);

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

      int[] cs = docs.getDocument(d).getChunks();

      for (int i=0; i<nd; i++) {

        int w = fs[i];
        int k = x[d][i];
        int c = cs[i];

        if (k == 1) {

          logProb += Math.log(getScore(w, r, c));

          incrementCounts(w, r, c);
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
          pw.println("#chunk register typeindex type proportion");
      }

      Probability[] probs = new Probability[W];

      for (int c=0; c<C; c++) {
        for (int r=0; r<R; r++) {
          for (int w=0; w<W; w++)
            probs[w] = new Probability(w, getScore(w, r, c));

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
              pw.print(c); pw.print(" ");
              pw.print(r); pw.print(" ");
              pw.print(probs[i].index); pw.print(" ");
              pw.print(dict.lookupObject(probs[i].index)); pw.print(" ");
              pw.print(probs[i].prob); pw.println();
            }
          }

          String string = line.toString();

          if ((fileName == null) || summary)
            if (!string.equals(""))
              pw.println("Chunk " + c + ", register " + r + ": " + string);
        }
      }
      if (fileName != null)
        pw.close();
    }
    catch (IOException e) {
      System.out.println(e);
    }
  }
}
