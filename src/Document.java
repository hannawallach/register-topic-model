package edu.umass.cs.wallach.cluster;

public class Document {

  private String source = null;

  private int[] tokens = null;
  private int[] chunks = null;

  private int register = -1;

  public Document(String source) {

    this.source = source;
  }

  public void setTokens(int[] tokens) {

    if (chunks != null)
      assert tokens.length == chunks.length;

    this.tokens = tokens;
  }

  public void setChunks(int[] chunks) {

    if (tokens != null)
      assert chunks.length == tokens.length;

    this.chunks = chunks;
  }

  public String getSource() {

    return source;
  }

  public int getLength() {

    return tokens.length;
  }

  public int getToken(int i) {

    return tokens[i];
  }

  public int[] getTokens() {

    return tokens;
  }

  public void setRegister(int register) {

    this.register = register;
  }

  public int getRegister() {

    return register;
  }

  public int getChunk(int i) {

    return chunks[i];
  }

  public int[] getChunks() {

    return chunks;
  }
}
