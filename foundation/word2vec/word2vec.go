//
// Copyright (C) 2024 Dmitry Kolesnikov
//
// This file may be modified and distributed under the terms
// of the MIT license.  See the LICENSE file for details.
// https://github.com/fogfish/word2vec
//

package word2vec

/*
#cgo CFLAGS: -Ilibw2v/lib
#cgo LDFLAGS: -L libw2v/lib -lw2v
#include <stdlib.h>
#include "libw2v/include/w2v.h"
*/
import "C"
import (
	"errors"
	"fmt"
	"unsafe"
)

// Nearest represents the word and the percent of closeness.
type Nearest struct {
	Word     string
	Distance float32
}

// =============================================================================

// Model represents a word2vec model.
type Model struct {
	fileModel  string
	vectorSize int
	h          unsafe.Pointer
}

// Loads takes a file on disk and loads it for processing.
func Load(fileModel string, vector int) (w2v Model, err error) {
	w2v.fileModel = fileModel
	w2v.vectorSize = 300
	if vector != 0 {
		w2v.vectorSize = vector
	}

	name := C.CString(w2v.fileModel)
	defer C.free(unsafe.Pointer(name))

	w2v.h = C.Load(name)
	if uintptr(w2v.h) == 0 {
		return w2v, fmt.Errorf("unable to load model")
	}

	return w2v, nil
}

// VectorOf calculates embedding vector for input term (word)
func (m *Model) VectorOf(word string, vector []float32) error {
	cword := C.CString(word)
	defer C.free(unsafe.Pointer(cword))

	ptr := C.VectorOf(m.h, cword)
	if ptr == nil {
		return errors.New("unknown tokens")
	}

	array := unsafe.Slice((*float32)(ptr), m.vectorSize)

	copy(vector, array)

	C.free(unsafe.Pointer(ptr))

	return nil
}

// Embedding calculates the embedding for document.
func (m *Model) Embedding(doc string, vector []float32) error {
	cdoc := C.CString(doc)
	defer C.free(unsafe.Pointer(cdoc))

	ptr := C.Embedding(m.h, cdoc)
	if ptr == nil {
		return errors.New("unknown tokens")
	}

	array := unsafe.Slice((*float32)(ptr), m.vectorSize)

	copy(vector, array)

	C.free(unsafe.Pointer(ptr))

	return nil
}

// Lookup nearest words from the model
func (m *Model) Lookup(query string, seq []Nearest) error {
	cq := C.CString(query)
	defer C.free(unsafe.Pointer(cq))

	type nearest_t struct {
		seq *C.float
		len C.ulong
		buf *C.char
	}

	k := len(seq)
	bag := (nearest_t)(C.Lookup(m.h, cq, C.ulong(k)))

	if bag.seq == nil || bag.buf == nil {
		return errors.New("unknown tokens")
	}

	seqd := unsafe.Slice((*float32)(bag.seq), k)
	seqw := unsafe.Slice((*C.char)(bag.buf), bag.len)

	p := 0
	for i := 0; i < k; i++ {
		seq[i].Distance = seqd[i]
		seq[i].Word = C.GoString(&seqw[p])
		p += len(seq[i].Word) + 1
	}

	C.free(unsafe.Pointer(bag.seq))
	C.free(unsafe.Pointer(bag.buf))

	return nil
}
