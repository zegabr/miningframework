#!/bin/bash
############################################################
# Help                                                     #
############################################################
helpText()
{
   # Display Help
   echo "Syntax: csdiff [-options] MYFILE OLDFILE YOURFILE"
   echo "options:"
   echo "-h                    Print this Help."
   echo "-s \"<separators>\"     Specify the list of separators, e.g. \"{ } ( ) ; ,\""
   echo
}

############################################################
# sed options used                                         #
############################################################
## Descriptions extracted from sed man page: https://linux.die.net/man/1/sed
# -e       - add the script to the commands to be executed
# :a       - Defines a label 'a'
# N        - Append the next line of input into the pattern space.
# $        - Match the last line.
# !        - After the address (or address-range), and before the command, a ! may be inserted, which specifies that the command shall only be executed if the address (or address-range) does not match.
# b[label] - Branch to [label]; if [label] is omitted, branch to end of script.
# s/       - Form: [s/regexp/replacement/] - Attempt to match regexp against the pattern space. If successful, replace that portion matched with replacement.

############################################################
############################################################
#######                   CSDiff                     #######
############################################################
############################################################
############################################################
# Process the input options. Add options as needed.        #
############################################################
while getopts s:h option
do
  case $option in
    h) # display Help
      helpText
      exit 0
      ;;
    s)
      set -f                 # turn off filename expansion
      separators=($OPTARG)   # variable is unquoted
      set +f                 # turn it back on
      ;;
   esac
done

shift $((OPTIND-1))

[ "${1:-}" = "--" ] && shift

############################################################
# Main logic                                               #
############################################################

parameters=("$@")
myFile=${parameters[0]}
oldFile=${parameters[1]}
yourFile=${parameters[2]}
parentFolder=$(echo "${myFile%/*}")
myFileBaseName="$(basename "${myFile}")"
fileExt=$([[ "$myFileBaseName" = *.* ]] && echo ".${myFileBaseName##*.}" || echo '')

myTempFile="${myFile}_temp${fileExt}"
oldTempFile="${oldFile}_temp${fileExt}"
yourTempFile="${yourFile}_temp${fileExt}"

add_dolar_sign_separators() {
    local inputFile="$1"
    awk -v separatorString="$(echo "$separators" | tr -d ' ')" '

      # loop through each line of input
    {
        # initialize output string
        output=""

        # loop through each character in the line
        for (i=1; i<=length($0); i++) {
            # check if the character is a separator
            if (index(separatorString, substr($0,i,1)) > 0) {
                # if separator is not at the end of the line, add \n$$$$$$$ before and after
                if (i < length($0)) {
                    output=output "\n$$$$$$$" substr($0,i,1) "\n$$$$$$$"
                }
                # if separator is at the end of the line, add \n$$$$$$$ before only
                else {
                    output=output "\n$$$$$$$" substr($0,i,1)
                }
            }
            # if character is not a separator, add it to output string
            else {
                output=output substr($0,i,1)
            }
        }

        # add \n$$$$$$$ to end of line if there is no separator at the end
        if (index(separatorString, substr(output,length(output),1)) == 0) {
            output=output "\n$$$$$$$"
        }

        # fix = character issue for some python files with big comments (this is undone in the end)
        gsub("=", "$=", output)
        # print the modified line
        print output
    }
    ' "$inputFile"
}

# Perform the tokenization of the input file based on the provided separators
add_dolar_sign_separators "$myFile" > "$myTempFile"
add_dolar_sign_separators "$yourFile" > "$yourTempFile"
add_dolar_sign_separators "$oldFile" > "$oldTempFile"
wait

# Runs diff3 against the tokenized inputs, generating a tokenized merged file
midMergedFile="${parentFolder}/mid_merged${fileExt}"
diff3 -m -E "$myTempFile" "$oldTempFile" "$yourTempFile" > $midMergedFile

# Removes the tokenized input files
rm "$myTempFile"
rm "$oldTempFile"
rm "$yourTempFile"
wait

# Removes the tokens from the merged file, generating the final merged file
mergedFile="${parentFolder}/merged${fileExt}"
awk '
  BEGIN {
    RS="\n\\$\\$\\$\\$\\$\\$\\$"
    ORS=""
  }
  {
      print
  }
' $midMergedFile > "$mergedFile".tmp && mv "$mergedFile".tmp "$mergedFile"
rm "$midMergedFile"
wait

# Get the names of left/base/right files
ESCAPED_LEFT=$(printf '%s\n' "${myFile}" | sed -e 's/[\/&]/\\&/g')
ESCAPED_BASE=$(printf '%s\n' "${oldFile}" | sed -e 's/[\/&]/\\&/g')
ESCAPED_RIGHT=$(printf '%s\n' "${yourFile}" | sed -e 's/[\/&]/\\&/g')

ESCAPED_TEMP_LEFT=$(printf '%s\n' "$myTempFile" | sed -e 's/[\/&]/\\&/g')
ESCAPED_TEMP_BASE=$(printf '%s\n' "$oldTempFile" | sed -e 's/[\/&]/\\&/g')
ESCAPED_TEMP_RIGHT=$(printf '%s\n' "$yourTempFile" | sed -e 's/[\/&]/\\&/g')

# # TODO: make this universal to other languages, here it will work with python
comment_string="#"

# # Fix the merged file line breaks that got messed up by the CSDiff algorithm.
sed -i -e "/^$comment_string/!s/\(<<<<<<< $ESCAPED_TEMP_LEFT\)\(.\+\)/\1\n\2/" $mergedFile
sed -i -e "/^$comment_string/!s/\(<<<<<<< $ESCAPED_TEMP_BASE\)\(.\+\)/\1\n\2/" $mergedFile
sed -i -e "/^$comment_string/!s/\(<<<<<<< $ESCAPED_TEMP_RIGHT\)\(.\+\)/\1\n\2/" $mergedFile
sed -i -e "/^$comment_string/!s/\(||||||| $ESCAPED_TEMP_BASE\)\(.\+\)/\1\n\2/" $mergedFile
sed -i -e "/^$comment_string/!s/\(||||||| $ESCAPED_TEMP_LEFT\)\(.\+\)/\1\n\2/" $mergedFile
sed -i -e "/^$comment_string/!s/\(||||||| $ESCAPED_TEMP_RIGHT\)\(.\+\)/\1\n\2/" $mergedFile
sed -i -e "/^$comment_string/!s/\(>>>>>>> $ESCAPED_TEMP_RIGHT\)\(.\+\)/\1\n\2/" $mergedFile
sed -i -e "/^$comment_string/!s/\(>>>>>>> $ESCAPED_TEMP_LEFT\)\(.\+\)/\1\n\2/" $mergedFile
sed -i -e "/^$comment_string/!s/\(>>>>>>> $ESCAPED_TEMP_BASE\)\(.\+\)/\1\n\2/" $mergedFile
sed -i -e '$a\ ' "$mergedFile"
wait


awk -v c_str="$comment_string" -v left="$ESCAPED_LEFT" -v base="$ESCAPED_BASE" -v right="$ESCAPED_RIGHT" \
  -v t_left="$ESCAPED_TEMP_LEFT" -v t_base="$ESCAPED_TEMP_BASE" -v t_right="$ESCAPED_TEMP_RIGHT" '
BEGIN {}
{
  if ($0 !~ "^" c_str) {
    gsub(t_left, left, $0)
    gsub(t_base, base, $0)
    gsub(t_right, right, $0)
  }
  print $0
}
' "$mergedFile" > "$mergedFile".tmp && wait && mv "$mergedFile".tmp "$mergedFile"

sed -i -e "/^$comment_string/!s/\(=======\)\(.\+\)/\1\n\2/" $mergedFile
sed -i -e :a -e '/^\n*$/{$d;N;ba' -e '}' "$mergedFile"
wait

# undoing replacement made at the beginning this should be made only after processing the "=======" strings above
awk '
{
  gsub("\\$=", "=", $0)
  print $0
}
' "$mergedFile" > "$mergedFile".tmp && wait && mv "$mergedFile".tmp "$mergedFile"

# remove last empty line (at some point in this script we are inserting a new one)
# TODO: find where we are inserting a new line to csdiff, treat this case ahd remove this code below
if [ -z "$(tail -c 1 "$mergedFile")" ]; then
    sed -i '$ d' "$mergedFile"
fi

mv "$mergedFile" "${parentFolder}/csdiff${fileExt}"

# Outputs two other files that will be useful for the study: one generated by the diff3 merge
# and another one generated by the 'git merge-file' command, using the diff3.
diff3 -E -m ${myFile} ${oldFile} ${yourFile} > "${parentFolder}/diff3${fileExt}"
git merge-file -p --diff3 ${myFile} ${oldFile} ${yourFile} > "${parentFolder}/git_merge${fileExt}"
