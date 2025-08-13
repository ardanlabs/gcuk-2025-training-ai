//
// Copyright (C) 2024 Dmitry Kolesnikov
//
// This file may be modified and distributed under the terms
// of the MIT license.  See the LICENSE file for details.
// https://github.com/fogfish/word2vec
//

package word2vec

/*
#include <stdlib.h>
#include "libw2v/include/w2v.h"
*/
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"
)

// ConfigCorpus represents the base config items.
type ConfigCorpus struct {
	// InputFile represents the file of the raw data to process.
	InputFile string

	// StopWordsFile represents the file of the stopwords to use.
	StopWordsFile string

	// Tokenizer represents the word delimiter.
	// Ex: " \n,.-!?:;/\"#$%&'()*+<=>@[]\\^_`{|}~\t\v\f\r"
	Tokenizer string

	// Sequencer represents the end of a sentence.
	// Ex: ".\n?!"
	Sequencer string
}

// ConfigWordVector represents word related config items.
type ConfigWordVector struct {
	// Vector represents the number of data points for the vector.
	// Ex: 300
	Vector int

	// Window represents the max skip length between words.
	// Ex: 5
	Window int

	// Threshold represents the occurrence of words. Those that appear with
	// higher frequency in the training data will be randomly down-sampled.
	// Ex: 1e-3
	Threshold float64

	// Frequency represents when words should be discarded that appear less
	// than <int> times.
	// Ex: 5
	Frequency int
}

// ConfigLearning represents learning related config items.
type ConfigLearning struct {
	// Epoch represents the number of training runs.
	// Ex: 10
	Epoch int

	// Rate represents the starting learning rate.
	// Ex: 0.05
	Rate float64
}

// Config defines the required setting for training.
type Config struct {
	Corpus   ConfigCorpus
	Vector   ConfigWordVector
	Learning ConfigLearning

	// choose of the learning model:
	//  - Continuous Bag of Words (CBOW)
	//  - Skip-Gram
	UseSkipGram bool
	UseCBOW     bool

	// the computationally efficient approximation
	//  - Negative Sampling (NS)
	//  - Hierarchical Softmax (HS)
	UseNegativeSampling    bool
	UseHierarchicalSoftMax bool

	// number of negative examples (NS option)
	SizeNegativeSampling int

	Output  string
	Threads int
	Verbose bool
}

// NewConfigDefault defines a set of default configuration options.
func NewConfigDefault() Config {
	return Config{
		Corpus: ConfigCorpus{
			Tokenizer: " \n,.-!?:;/\"#$%&'()*+<=>@[]\\^_`{|}~\t\v\f\r",
			Sequencer: ".\n?!",
		},
		Vector: ConfigWordVector{
			Vector:    300,
			Window:    10,
			Threshold: 1e-3,
			Frequency: 5,
		},
		Learning: ConfigLearning{
			Epoch: 5,
			Rate:  0.05,
		},
		UseSkipGram:            true,
		UseCBOW:                false,
		UseNegativeSampling:    true,
		UseHierarchicalSoftMax: false,
		SizeNegativeSampling:   5,
		Threads:                runtime.GOMAXPROCS(0),
		Verbose:                true,
	}
}

// =============================================================================

// Train performs a training run and produces a new model.
func Train(config Config) error {
	w2v := struct {
		config Config
		h      unsafe.Pointer
	}{
		config: config,
	}

	dataset := C.CString(w2v.config.Corpus.InputFile)
	defer C.free(unsafe.Pointer(dataset))

	fileStopWords := C.CString(w2v.config.Corpus.StopWordsFile)
	defer C.free(unsafe.Pointer(fileStopWords))

	fileModel := C.CString(w2v.config.Output)
	defer C.free(unsafe.Pointer(fileModel))

	withHS := C.uchar(0)
	if w2v.config.UseHierarchicalSoftMax {
		withHS = C.uchar(1)
	}

	withSG := C.uchar(0)
	if w2v.config.UseSkipGram {
		withSG = C.uchar(1)
	}

	tokenizer := C.CString(w2v.config.Corpus.Tokenizer)
	defer C.free(unsafe.Pointer(tokenizer))

	sequencer := C.CString(w2v.config.Corpus.Sequencer)
	defer C.free(unsafe.Pointer(sequencer))

	verbose := C.uchar(0)
	if w2v.config.Verbose {
		verbose = C.uchar(1)
	}

	w2v.h = C.Train(
		dataset,
		fileStopWords,
		fileModel,
		C.ushort(w2v.config.Vector.Frequency),
		C.ushort(w2v.config.Vector.Vector),
		C.uchar(w2v.config.Vector.Window),
		C.float(w2v.config.Vector.Threshold),
		withHS,
		C.uint8_t(w2v.config.SizeNegativeSampling),
		C.uint8_t(w2v.config.Threads),
		C.uint8_t(w2v.config.Learning.Epoch),
		C.float(w2v.config.Learning.Rate),
		withSG,
		tokenizer,
		sequencer,
		verbose,
	)

	if uintptr(w2v.h) == 0 {
		return fmt.Errorf("unable to train model")
	}

	return nil
}
