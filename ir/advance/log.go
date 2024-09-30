package advance

import (
	"log"
	"os"
)

func SetupLogger(filename, classname, level string) *log.Logger {
	logger := log.New(os.Stdout, "", log.LstdFlags)
	logger.SetPrefix(filename + "." + classname + " ")
	switch level {
	case "DEBUG":
		logger.SetFlags(log.LstdFlags | log.Lshortfile)
	case "INFO":
		logger.SetFlags(log.LstdFlags)
	case "WARN":
		logger.SetFlags(log.LstdFlags)
	case "ERROR":
		logger.SetFlags(log.LstdFlags | log.Lshortfile)
	case "CRITICAL":
		logger.SetFlags(log.LstdFlags | log.Lshortfile)
	default:
		logger.SetFlags(log.LstdFlags)
	}
	return logger
}
