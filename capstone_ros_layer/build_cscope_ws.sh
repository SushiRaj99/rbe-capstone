#!/bin/bash

code_src=""
database_dir=""
gvim_server=""
ignore_exprs=""
find_exprs=""
bypass_init=0
part2full_ratio=10

# Configure script variables based on user inputs:
while [ $# -ge 1 ]; do
	case $1 in
		'-s')	# assign source code directory to build cscope/ctags database from
			shift
			code_src="${1}"
			;;
		'-d')	# assign destination for database directory
			shift
			database_dir="${1}"
			;;
		'-f')	# assign expressions for files to find and include within cscope database management
			shift
			find_exprs="${1}"
			;;
		'-g')	# assign gvim server name
			shift
			gvim_server="${1}"
			;;
		'-i')	# defines expressions for directories/files to ignore within cscope database management
			shift
			ignore_exprs="${1}"
			;;
		'-b')	# flag to bypass initial database creation (i.e. "initialization")
			bypass_init=1
			;;
		'-p2f')	# defines number of partial refreshes of the cscope database for every full refreshes
			shift
			part2full_ratio="${1}"
			;;
		'-h')	# help
			echo "build_cscope_ws() contains the following input argument options:"
			echo -e "\t-s\n\t\t[Arguments: 1][Required] Specifies the code source directories (space delimited) to build cscope and ctags database from." | fmt
			echo -e "\t-d\n\t\t[Arguments: 1][Optional] Specifies the destination directory where the cscope and ctags files will be generated." | fmt
			echo -e "\t-f\n\t\t[Arguments: 1][Required] Specifies expressions (space delimited) for finding files to include within the cscope and ctags database." | fmt
			echo -e "\t-g\n\t\t[Arguments: 1][Optional] Specifies the gvim servername." | fmt
			echo -e "\t-i\n\t\t[Arguments: 1][Optional] Specifies expressions (space delimited) for ignoring or pruning files to include within the cscope and ctags database." | fmt
			echo -e "\t-b\n\t\t[Arguments: 0][Optional] Specifies flag to bypass initial cscope and ctags database generation (in case databases already exist)." | fmt
			echo -e "\t-p2f\n\t\t[Arguments: 1][Optional] Specifies the number of 'partial' refreshes for every 'full' refresh of the cscope and ctags databases." | fmt
			exit
			;;
		*)
			echo "ERROR: Invalid input argument. Seek help via 'build_cscope_ws.sh -h'"
			exit
			;;
	esac
	shift;
done

if [ -z "${code_src}" ] || [ -z "${find_exprs}" ]; then
	echo "ERROR: Invalid input format. Seek help via 'build_cscope_ws.sh -h'"
	exit
elif [ -z "${database_dir}" ]; then
	tmp_code_srcarray=("${code_src}")
	database_dir="${tmp_code_srcarray[0]}/cscope_ws"
fi
if [ -z "${gvim_server}" ]; then
	gvim_server="GVIM_CSCOPE_SERVER"
fi
expanded_findexpr="${find_exprs}"
# Obsolete code kept as comments for history:
#if [ ! -z "${appended_finds}" ]; then
#	expanded_findexpr+=" ${appended_finds}"
#fi
expanded_ignrexpr="${ignore_exprs}"
# Obsolete code kept as comments for history:
#if [ ! -z "${appended_ignores}" ]; then
#	expanded_ignrexpr+=" ${appended_ignores}"
#fi
expanded_ignrexpr+=" ${database_dir}"

# Initialize local functions:
create_launch_files(){
	local gvim_exe="${1}"
	local cscope_exe="${2}"
	local vimserver="${3}"
	echo "#!/bin/bash" > "${gvim_exe}"
	echo "" >> "${gvim_exe}"
	echo "servercheck=\$(gvim --serverlist | grep -ho \"${vimserver}\")" >> "${gvim_exe}"
	echo "if [ ! -z \"\${servercheck}\" ]; then" >> "${gvim_exe}"
	echo "   filename=\"\$@\"" >> "${gvim_exe}"
        echo "   filedir=(\${filename})" >> "${gvim_exe}"
	echo "   filedir=\"\${filedir[\$(( \${#filedir[@]} - 1 ))]%/*}\"" >> "${gvim_exe}"
	echo "   filename=\"\${filename##*/}\"" >> "${gvim_exe}"
	echo "   swpfile=\"\${filedir}/.\${filename}.swp\"" >> "${gvim_exe}"
	echo "   if [ ! -f \${swpfile} ]; then" >> "${gvim_exe}"	
	echo "      gvim --servername ${vimserver} --remote-tab \"\$@\"" >> "${gvim_exe}"
	echo "      gvim --servername ${vimserver} --remote-send \":set number<CR>\"" >> "${gvim_exe}"
	echo "      gvim --servername ${vimserver} --remote-send \":Vex<CR>\"" >> "${gvim_exe}"
	echo "      gvim --servername ${vimserver} --remote-send \"<C-W><Right>\"" >> "${gvim_exe}"
	echo "      gvim --servername ${vimserver} --remote-send \"zR\"" >> "${gvim_exe}"
	echo "   fi" >> "${gvim_exe}"
	echo "else" >> "${gvim_exe}"
	echo "   gvim -Nu /etc/vim/vimrc --servername ${vimserver} \"\$@\"" >> "${gvim_exe}"	# The "$@" allows filenames to be passed in as arguments
	echo "   gvim --servername ${vimserver} --remote-send \":let g:netrw_banner = 0<CR>\"" >> "${gvim_exe}"
	echo "   gvim --servername ${vimserver} --remote-send \":let g:netrw_liststyle = 3<CR>\"" >> "${gvim_exe}"
	echo "   gvim --servername ${vimserver} --remote-send \":let g:netrw_browse_split = 4<CR>\"" >> "${gvim_exe}"
	echo "   gvim --servername ${vimserver} --remote-send \":let g:netrw_altv = 1<CR>\"" >> "${gvim_exe}"
	echo "   gvim --servername ${vimserver} --remote-send \":let g:netrw_winsize = 25<CR>\"" >> "${gvim_exe}"
	echo "   gvim --servername ${vimserver} --remote-send \":function! Reset_Cscope()<CR>cscope reset<CR>endfunction<CR><CR>\"" >> "${gvim_exe}"
	echo "   gvim --servername ${vimserver} --remote-send \":colorscheme desert<CR>\"" >> "${gvim_exe}"
	echo "   gvim --servername ${vimserver} --remote-send \":set nowrap<CR>\"" >> "${gvim_exe}"
	echo "   gvim --servername ${vimserver} --remote-send \":set guioptions+=b<CR>\"" >> "${gvim_exe}"
	echo "   gvim --servername ${vimserver} --remote-send \":set number<CR>\"" >> "${gvim_exe}"
	echo "   gvim --servername ${vimserver} --remote-send \":Vex<CR>\"" >> "${gvim_exe}"
	echo "   gvim --servername ${vimserver} --remote-send \":set mouse=a<CR>\"" >> "${gvim_exe}"
	echo "   gvim --servername ${vimserver} --remote-send \"<CR>\"" >> "${gvim_exe}"
	echo "   gvim --servername ${vimserver} --remote-send \"<C-W><Right>\"" >> "${gvim_exe}"
	echo "   gvim --servername ${vimserver} --remote-send \":set foldmethod=syntax<CR>\"" >> "${gvim_exe}"
	echo "   gvim --servername ${vimserver} --remote-send \":set foldcolumn=3<CR>\"" >> "${gvim_exe}"
	echo "   gvim --servername ${vimserver} --remote-send \"zR\"" >> "${gvim_exe}"
	echo "fi" >> "${gvim_exe}"
	echo "#!/bin/bash" > "${cscope_exe}"
	echo "" >> "${cscope_exe}"
	echo "export CSCOPE_EDITOR=\"${gvim_exe}\"" >> "${cscope_exe}"
	echo "cscope -d" >> "${cscope_exe}"
	chmod 755 "${gvim_exe}" "${cscope_exe}"
}

partial_refresh(){
	local srcdir="${1}"
	local destdir="${2}"
	local inclexpr="${3}"
	local ignrexpr="${4}"
	local tagfile=$"${destdir}/tags"
	local ctlog="${destdir}/ctags.log"
	local dbfile_mod="${destdir}/cscope.files_mod"
	local inclargs=(${inclexpr})
	local ignrargs=(${ignrexpr})
	local inclfullargs=()
	local ignrfullargs=()
	local srcdirarr=(${srcdir})
	for expr in ${inclargs[*]}; do
		[[ -z "${inclfullargs}" ]] && inclfullargs+=(-name "${expr}") || inclfullargs+=( -o -name "${expr}")
	done
	for expr in ${ignrargs[*]}; do
		ignrfullargs+=(-o -wholename "${expr}" -prune)
	done
	> "${dbfile_mod}"	# null create the new dbfile and append to it via find
	for single_srcdir in ${srcdirarr[*]}; do
		find "${single_srcdir}/" -type f -newer "${tagfile}" ${inclfullargs[@]} ${ignrfullargs[@]} >> "${dbfile_mod}"
	done
	ctags --append -L "${dbfile_mod}" -f "${tagfile}" > "${ctlog}" 2> "${ctlog}"
}

full_refresh(){
	local srcdir="${1}"
	local destdir="${2}"
	local inclexpr="${3}"
	local ignrexpr="${4}"
	local gserver="${5}"
	local dbfile="${destdir}/cscope.files"
	local tagfile=$"${destdir}/tags"
	local cslog="${destdir}/cscope.log"
	local ctlog="${destdir}/ctags.log"
	local inclargs=(${inclexpr})
	local ignrargs=(${ignrexpr})
	local inclfullargs=()
	local ignrfullargs=()
	local tagfile_tmp="${tagfile}_tmp"
	local srcdirarr=(${srcdir})
	for expr in ${inclargs[*]}; do
		[[ -z "${inclfullargs}" ]] && inclfullargs+=(-name "${expr}") || inclfullargs+=( -o -name "${expr}")
	done
	for expr in ${ignrargs[*]}; do
		ignrfullargs+=(-o -wholename "${expr}" -prune)
	done
	# Debugging:
	#echo "inclfullargs[*] = ${inclfullargs[*]}"
	#echo "ignrfullargs[*] = ${ignrfullargs[*]}"
	> "${dbfile}"	# null create the new dbfile and append to it via find
	for single_srcdir in ${srcdirarr[*]}; do
		find "${single_srcdir}/" ${inclfullargs[@]} ${ignrfullargs[@]} >> "${dbfile}"
	done
	pushd "${destdir}" > /dev/null
	cscope -b -q -k -i "${dbfile}" > "${cslog}" 2> "${cslog}"
	popd > /dev/null
	ctags -L "${dbfile}" -f "${tagfile_tmp}" > "${ctlog}" 2> "${ctlog}"
	mv -f "${tagfile_tmp}" "${tagfile}" > /dev/null 2> /dev/null
	# If the dedicated gvim server is active, reset the cscope database that's been loaded into gvim:
	[[ $(gvim --serverlist) == *"${gserver}"* ]] && gvim --servername "${gserver}" --remote-expr "Reset_Cscope()" > /dev/null 2> /dev/null
}

# Perform actual processing:
if [ ! -d "${database_dir}" ]; then
	mkdir -p "${database_dir}"
fi
gvimlaunchfile="${database_dir}/launch_gvim.sh"
cscopelaunchfile="${database_dir}/launch_cscope_ui.sh"
create_launch_files "${gvimlaunchfile}" "${cscopelaunchfile}" "${gvim_server}"

if [[ ${bypass_init} -le 0 ]]; then
	echo "Creating cscope and ctags databases for ${code_src} in ${database_dir}..."
	full_refresh "${code_src}" "${database_dir}" "${expanded_findexpr}" "${expanded_ignrexpr}" "${gvim_server}"
	echo ""
	echo "Done creating initial database."
fi
echo ""
echo "Navigate to ${database_dir} and call 'launch_cscope_ui.sh' in a new terminal window."
echo ""
echo "Leave this window open and running to continually update database files in the background..."

loopcnt=0
while true; do
	if [[ ${loopcnt} -lt ${part2full_ratio} ]]; then
		partial_refresh "${code_src}" "${database_dir}" "${expanded_findexpr}" "${expanded_ignrexpr}"
		(( loopcnt++ ))
	else	
		full_refresh "${code_src}" "${database_dir}" "${expanded_findexpr}" "${expanded_ignrexpr}" "${gvim_server}"
		loopcnt=0
	fi
	sleep 1m
done
