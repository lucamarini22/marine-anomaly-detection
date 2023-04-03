class FileIO:
    def append(file_path: str, text: str):
        """Appends text to a file.

        Args:
            file_path (str): path of the file.
            text (str): text to append.
        """
        with open(file_path, "a") as myfile:
            myfile.write(text)
