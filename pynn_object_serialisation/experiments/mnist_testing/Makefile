RESULT_ROOT_DIR=./results
RESULT_DIRS=$(wildcard $(RESULT_ROOT_DIR)/*/)
RESULT_SUMMARIES=$(wildcard $(RESULT_ROOT_DIR)/*/summary.csv)

## all	: Generate plot of accuracy
.PHONY : all
plot.png : master_summary.csv
	python summary_plotter.py $^

## summaries	: Generate summaries for a results directories
.PHONY : summaries
summaries : $(RESULT_DIRS)summary.csv

#Collates summaries into one master_summary
master_summary.csv: $(RESULT_ROOT_DIR) summaries
	python summary_collator.py $<

%/summary.csv :
	python data_summary_generator.py $*

## variables	: List variables
.PHONY : variables
variables:
	@echo RESULT_ROOT_DIR: $(RESULT_ROOT_DIR)
	@echo RESULT_DIRS: $(RESULT_DIRS)
	@echo RESULT_SUMMARIES: $(RESULT_SUMMARIES)

## clean       : Remove auto-generated files.
.PHONY : clean
clean :
	rm -f plot.png
	rm -f master_summary.csv
	rm -f $(RESULT_SUMMARIES)

.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<
