BUILD_DIR = build
LIBS_DIR = libs
SRC_DIR = src
DATA_DIR = data
RESULTS_DIR = results
SCRIPTS_DIR = scripts

MAX_HEAP = 500m

JAVA_FLAGS = -server -enableassertions -Xmx$(MAX_HEAP) -XX:MaxPermSize=500m

CP = $(BUILD_DIR):$(LIBS_DIR)/mallet.jar:$(LIBS_DIR)/mallet-deps.jar

# by default simply compile source code

all: $(BUILD_DIR)

.PHONY: $(BUILD_DIR)

# compilation is handled by ant

$(BUILD_DIR): #clean
	ant build

# experiments...

.PRECIOUS: $(DATA_DIR)/patents/%

$(DATA_DIR)/patents/%: $(DATA_DIR)/patents/%.tar.gz
	tar zxvf $< -C $(@D)

$(DATA_DIR)/patents/%_no_stopwords.dat: $(DATA_DIR)/patents/%
	java $(JAVA_FLAGS) \
	-classpath $(CP) \
	cc.mallet.classify.tui.Text2Vectors \
	--keep-sequence \
	--remove-stopwords \
	--extra-stopwords $(DATA_DIR)/stopwordlist.txt \
	--output $@ \
	--input $<

$(DATA_DIR)/patents/%.dat: $(DATA_DIR)/patents/%
	java $(JAVA_FLAGS) \
	-classpath $(CP) \
	cc.mallet.classify.tui.Text2Vectors \
	--keep-sequence \
	--output $@ \
	--input $<

.PHONY: merge

merge:
	java $(JAVA_FLAGS) \
	-classpath $(CP) \
	edu.umass.cs.wallach.cluster.ChunkInstanceListMerger \
	$(DATA_DIR)/patents/core.dat \
	$(DATA_DIR)/patents/core_with_chunks.dat

$(RESULTS_DIR)/lda/%/T$(T)-S$(S)-SAMPLE$(SAMPLE)-ID$(ID):
	mkdir -p $@; \
	I=`expr $(S) / 10`; \
	java $(JAVA_FLAGS) \
	-classpath $(CP) \
        edu.umass.cs.wallach.cluster.LDAExperiment \
	$(DATA_DIR)/patents/$*.dat \
	$(T) \
	$(S) \
	20 \
	$$I \
	$(SAMPLE) \
	$@ \
	> $@/stdout.txt

$(RESULTS_DIR)/background_lda/%/T$(T)-S$(S)-SAMPLE$(SAMPLE)-ID$(ID):
	mkdir -p $@; \
	I=`expr $(S) / 10`; \
	java $(JAVA_FLAGS) \
	-classpath $(CP) \
        edu.umass.cs.wallach.cluster.BackgroundLDAExperiment \
	$(DATA_DIR)/patents/$*.dat \
	$(T) \
	$(S) \
	20 \
	$$I \
	$(SAMPLE) \
	$@ \
	> $@/stdout.txt

$(RESULTS_DIR)/register_lda/%/T$(T)-R$(R)-S$(S)-SAMPLE$(SAMPLE)-ID$(ID):
	mkdir -p $@; \
	I=`expr $(S) / 10`; \
	java $(JAVA_FLAGS) \
	-classpath $(CP) \
	edu.umass.cs.wallach.cluster.RegisterLDAExperiment \
	$(DATA_DIR)/patents/$*.dat \
	$(T) \
	$(R) \
	$(S) \
	20 \
	$$I \
	$(SAMPLE) \
	$@ \
	> $@/stdout.txt

$(RESULTS_DIR)/chunk_register_lda/%/T$(T)-R$(R)-C$(C)-S$(S)-SAMPLE$(SAMPLE)-ID$(ID):
	mkdir -p $@; \
	I=`expr $(S) / 10`; \
	java $(JAVA_FLAGS) \
	-classpath $(CP) \
	edu.umass.cs.wallach.cluster.ChunkRegisterLDAExperiment \
	$(DATA_DIR)/patents/$*.dat \
	$(T) \
	$(R) \
	$(C) \
	$(S) \
	20 \
	$$I \
	$(SAMPLE) \
	$@ \
	> $@/stdout.txt

clean:
	ant clean
