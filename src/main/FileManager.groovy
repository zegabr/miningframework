final class FileManager {

    public static Set<String> getModifiedFiles(Project project, String childSHA, String ancestorSHA) {
        Set<String> modifiedFiles = new HashSet<String>()
        Process gitDiff = new ProcessBuilder('git', 'diff', '--name-only', childSHA, ancestorSHA)
            .directory(new File(project.getPath()))
            .start()
        
        gitDiff.getInputStream().eachLine {
            if(it.endsWith('.java'))
                modifiedFiles.add(it)
        }

        return modifiedFiles
    }

    public static File copyFile(Project project, String path, String SHA) {
        Process gitCatFile = new ProcessBuilder('git', 'cat-file', '-p', "${SHA}:${path}")
            .directory(new File(project.getPath()))
            .start()
    
        
        File target = new File("${SHA}.java")
        gitCatFile.getInputStream().eachLine {
            target << "${it}\n"
        }
       
        return target
    }

    public static delete(File file) {
        if (!file.isDirectory())
            file.delete()
        else {
            if (file.list().length == 0) 
                file.delete()
            else {
                String[] files = file.list()
                for (temp in files) {
                    delete(new File(file, temp))
                }
                if (file.list().length == 0) 
                    file.delete()
            }
        }
    }

}