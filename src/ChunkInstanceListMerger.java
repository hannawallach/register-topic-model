package edu.umass.cs.wallach.cluster;

import java.io.*;
import java.util.*;

import gnu.trove.*;

import cc.mallet.types.*;

public class ChunkInstanceListMerger {

  public static void main(String[] args) {

    if (args.length < 2) {
      System.out.println("Usage: ChunkInstanceListMerger <chunk_instance_list> <chunk_instance_list> ... <output_instance_list>");
      System.exit(1);
    }

    int numChunks = args.length - 1;

    String outputInstanceListFileName = args[numChunks];

    String[] files = new String[numChunks];
    System.arraycopy(args, 0, files, 0, numChunks);

    InstanceList mergedList = null;

    HashMap<String, Instance[]> nameToInstances = new HashMap<String, Instance[]>();

    System.out.print("About to merge " + numChunks + " chunks...");

    for (int c=0; c<numChunks; c++) {

      InstanceList list = InstanceList.load(new File(files[c]));

      if (c == 0)
        mergedList = list.cloneEmpty();

      for (Instance instance : list) {

        String name = (String) instance.getName().toString();

        if (!nameToInstances.containsKey(name))
          nameToInstances.put(name, new Instance[numChunks]);

        nameToInstances.get(name)[c] = instance;
      }
    }

    // create merged InstanceList -- note that the instances will NOT
    // be in the same order as in original InstanceLists...

    for (String name : nameToInstances.keySet()) {

      Instance[] instances = nameToInstances.get(name);

      Instance mergedInstance = null;

      TIntArrayList chunks = new TIntArrayList();

      assert instances.length == numChunks;

      for (int c=0; c<numChunks; c++) {

        Instance instance = instances[c];

        if (instance == null)
          continue;

        if (mergedInstance == null) {

          mergedInstance = new Instance(new FeatureSequence(instance.getAlphabet()), instance.getTarget(), instance.getName(), instance.getSource());

          if (mergedInstance.getSource() == null)
            mergedInstance.setSource(instance.getName());
        }

        FeatureSequence fs = (FeatureSequence) instance.getData();

        for (int i=0; i<fs.getLength(); i++) {

          ((FeatureSequence) mergedInstance.getData()).add(fs.get(i));

          // should check whether the target, source, etc. are the same

          chunks.add(c);
        }
      }

      mergedInstance.setProperty("chunks", chunks.toNativeArray());
      mergedList.add(mergedInstance);
    }

    System.out.println();

    System.out.print("Saving " + mergedList.size() + " instances to " + outputInstanceListFileName + "...");

    mergedList.save(new File(outputInstanceListFileName));

    System.out.println();
  }
}
