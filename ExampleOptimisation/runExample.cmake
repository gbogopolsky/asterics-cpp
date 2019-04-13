add_custom_target(run_all)
add_custom_target(plot_all)

add_dependencies(plot_all run_all)

set(OUTPUT_PERF_DIR "${CMAKE_BINARY_DIR}/Examples/Performances")

# Run the given target
# 	targetName : name of the target to be runned
function(runExample targetName)
	add_custom_command(OUTPUT ${OUTPUT_PERF_DIR}/${targetName}.txt
		COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${targetName} 2> ${OUTPUT_PERF_DIR}/${targetName}.txt
		WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
		COMMENT "Run ${targetName} program"
		DEPENDS ${targetName}
	)
	add_custom_target("run_${targetName}"  DEPENDS ${OUTPUT_PERF_DIR}/${targetName}.txt)
	add_dependencies("run_${targetName}" ${targetName})
	add_dependencies(run_all "run_${targetName}")
endfunction(runExample)

# Run the given python script
# 	scriptName : name of the script to be ran
# 	installModuleDependency : dependency of the script
function(runPythonExample scriptName installModuleDependency)
	get_filename_component(targetName ${scriptName} NAME_WE)
	
	add_custom_command(OUTPUT ${OUTPUT_PERF_DIR}/${targetName}.txt
		COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${scriptName} 2> ${OUTPUT_PERF_DIR}/${targetName}.txt
		WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
		COMMENT "Run ${PYTHON_EXECUTABLE} ${scriptName} program with target ${targetName}"
		DEPENDS ${scriptName}
	)
	add_custom_target("run_${targetName}"  DEPENDS ${OUTPUT_PERF_DIR}/${targetName}.txt)
	add_dependencies("run_${targetName}" ${installModuleDependency})
	add_dependencies(run_all "run_${targetName}")
endfunction(runPythonExample)

# Plot the performances of the different output
# baseOutputPng : base name of the png output file
# ARGN : list of target to be compared
function(plotPerf baseOutputPng)
	set(GNUPLOT_FILE "${OUTPUT_PERF_DIR}/${baseOutputPng}.gnuplot")
	file(WRITE ${GNUPLOT_FILE} "set terminal png notransparent crop enhanced size 800,600 font \"arial,14\"\n")
	file(APPEND ${GNUPLOT_FILE} "set grid xtics ytics mytics\n")
	file(APPEND ${GNUPLOT_FILE} "set key bottom right\n")
	file(APPEND ${GNUPLOT_FILE} "set logscale y\n")
	file(APPEND ${GNUPLOT_FILE} "set xlabel \"nb elements\"\n")
	file(APPEND ${GNUPLOT_FILE} "set ylabel \"elapsed time per element [cy/el]\"\n")
	file(APPEND ${GNUPLOT_FILE} "set output \"${baseOutputPng}ElapsedTimeCyEl.png\"\n")
	file(APPEND ${GNUPLOT_FILE} "plot ")
	
	set(listDepend)
	foreach(inputTarget ${ARGN})
		string(REPLACE "_" " " legendStr ${inputTarget})
		file(APPEND ${GNUPLOT_FILE} "\"${inputTarget}.txt\" using 1:2 title \"${legendStr}\" with lines  lw 2,")
		list(APPEND listDepend "${OUTPUT_PERF_DIR}/${inputTarget}.txt")
	endforeach(inputTarget)
	file(APPEND ${GNUPLOT_FILE} "\n")
	
	file(APPEND ${GNUPLOT_FILE} "set xlabel \"nb elements\"\n")
	file(APPEND ${GNUPLOT_FILE} "set ylabel \"elapsed time [cy]\"\n")
	file(APPEND ${GNUPLOT_FILE} "set output \"${baseOutputPng}ElapsedTime.png\"\n")
	file(APPEND ${GNUPLOT_FILE} "plot ")
	
	foreach(inputTarget ${ARGN})
		string(REPLACE "_" " " legendStr ${inputTarget})
		file(APPEND ${GNUPLOT_FILE} "\"${inputTarget}.txt\" using 1:3 title \"${legendStr}\" with lines  lw 2,")
	endforeach(inputTarget)
	file(APPEND ${GNUPLOT_FILE} "\n")
	
	add_custom_command(OUTPUT ${OUTPUT_PERF_DIR}/${baseOutputPng}ElapsedTimeCyEl.png ${OUTPUT_PERF_DIR}/${baseOutputPng}ElapsedTime.png
		COMMAND gnuplot  ${GNUPLOT_FILE}
		WORKING_DIRECTORY "${OUTPUT_PERF_DIR}"
		COMMENT "Call gnuplot ${baseOutputPng}"
		DEPENDS ${listDepend}
	)
	add_custom_target("plot_${baseOutputPng}"  DEPENDS ${OUTPUT_PERF_DIR}/${baseOutputPng}ElapsedTimeCyEl.png ${OUTPUT_PERF_DIR}/${baseOutputPng}ElapsedTime.png)
	
	foreach(inputTarget ${ARGN})
		add_dependencies("plot_${baseOutputPng}" "run_${inputTarget}")
	endforeach(inputTarget)
	
	
	add_dependencies(plot_all "plot_${baseOutputPng}")
endfunction(plotPerf)

# Plot the performances of the different output with probability on X axis and no more nb elements
# baseOutputPng : base name of the png output file
# ARGN : list of target to be compared
function(plotPerfProba baseOutputPng)
	set(GNUPLOT_FILE "${OUTPUT_PERF_DIR}/${baseOutputPng}.gnuplot")
	file(WRITE ${GNUPLOT_FILE} "set terminal png notransparent crop enhanced size 800,600 font \"arial,14\"\n")
	file(APPEND ${GNUPLOT_FILE} "set grid xtics ytics mytics\n")
	file(APPEND ${GNUPLOT_FILE} "set key bottom right\n")
	file(APPEND ${GNUPLOT_FILE} "set logscale y\n")
	file(APPEND ${GNUPLOT_FILE} "set xlabel \"proba\"\n")
	file(APPEND ${GNUPLOT_FILE} "set ylabel \"elapsed time per element [cy/el]\"\n")
	file(APPEND ${GNUPLOT_FILE} "set output \"${baseOutputPng}ElapsedTimeCyEl.png\"\n")
	file(APPEND ${GNUPLOT_FILE} "plot ")
	
	set(listDepend)
	foreach(inputTarget ${ARGN})
		string(REPLACE "_" " " legendStr ${inputTarget})
		file(APPEND ${GNUPLOT_FILE} "\"${inputTarget}.txt\" using 1:2 title \"${legendStr}\" with lines  lw 2,")
		list(APPEND listDepend "${OUTPUT_PERF_DIR}/${inputTarget}.txt")
	endforeach(inputTarget)
	file(APPEND ${GNUPLOT_FILE} "\n")
	
	file(APPEND ${GNUPLOT_FILE} "set xlabel \"proba\"\n")
	file(APPEND ${GNUPLOT_FILE} "set ylabel \"elapsed time [cy]\"\n")
	file(APPEND ${GNUPLOT_FILE} "set output \"${baseOutputPng}ElapsedTime.png\"\n")
	file(APPEND ${GNUPLOT_FILE} "plot ")
	
	foreach(inputTarget ${ARGN})
		string(REPLACE "_" " " legendStr ${inputTarget})
		file(APPEND ${GNUPLOT_FILE} "\"${inputTarget}.txt\" using 1:3 title \"${legendStr}\" with lines  lw 2,")
	endforeach(inputTarget)
	file(APPEND ${GNUPLOT_FILE} "\n")
	
	add_custom_command(OUTPUT ${OUTPUT_PERF_DIR}/${baseOutputPng}ElapsedTimeCyEl.png ${OUTPUT_PERF_DIR}/${baseOutputPng}ElapsedTime.png
		COMMAND gnuplot  ${GNUPLOT_FILE}
		WORKING_DIRECTORY "${OUTPUT_PERF_DIR}"
		COMMENT "Call gnuplot ${baseOutputPng}"
		DEPENDS ${listDepend}
	)
	add_custom_target("plot_${baseOutputPng}"  DEPENDS ${OUTPUT_PERF_DIR}/${baseOutputPng}ElapsedTimeCyEl.png ${OUTPUT_PERF_DIR}/${baseOutputPng}ElapsedTime.png)
	foreach(inputTarget ${ARGN})
		add_dependencies("plot_${baseOutputPng}" "run_${inputTarget}")
	endforeach(inputTarget)
	add_dependencies(plot_all "plot_${baseOutputPng}")
endfunction(plotPerfProba)

