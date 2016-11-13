//
// This script converts the CIFAR dataset to the leveldb format used
// by caffe to perform classification.
// Usage:
//    convert_cifar_data input_folder output_db_file
// The CIFAR dataset could be downloaded at
//    http://www.cs.toronto.edu/~kriz/cifar.html

#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "boost/scoped_ptr.hpp"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "stdint.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"

using caffe::Datum;
using boost::scoped_ptr;
using std::string;
namespace db = caffe::db;

const int kCIFARSize = 32;
const int kCIFARImageNBytes = 3072;
const int kCIFARBatchSize = 10000;
const int kCIFARTrainBatches = 5;

void read_image(std::ifstream* file, int* label, char* buffer) {
  char label_char;
  file->read(&label_char, 1);
  *label = label_char;
  file->read(buffer, kCIFARImageNBytes);
  return;
}

void convert_dataset(const string& input_folder, const string& output_folder,
    const string& db_type, const int pad) {
  scoped_ptr<db::DB> train_db(db::GetDB(db_type));
  train_db->Open(output_folder + "/cifar10_train_" + db_type, db::NEW);
  scoped_ptr<db::Transaction> txn(train_db->NewTransaction());
  // Data buffer
  int label;
  char str_buffer[kCIFARImageNBytes];
  int padded_imange_size = kCIFARSize+2*pad;
  char *str_buffer_pad = (char *)malloc(padded_imange_size*padded_imange_size*3);
  memset(str_buffer_pad,0,padded_imange_size*padded_imange_size*3);
  Datum datum;
  datum.set_channels(3);
  //datum.set_height(kCIFARSize);
  //datum.set_width(kCIFARSize);
  datum.set_height(padded_imange_size);
  datum.set_width(padded_imange_size);

  LOG(INFO) << "Writing Training data";
  for (int fileid = 0; fileid < kCIFARTrainBatches; ++fileid) {
    // Open files
    LOG(INFO) << "Training Batch " << fileid + 1;
    string batchFileName = input_folder + "/data_batch_"
      + caffe::format_int(fileid+1) + ".bin";
    std::ifstream data_file(batchFileName.c_str(),
        std::ios::in | std::ios::binary);
    CHECK(data_file) << "Unable to open train file #" << fileid + 1;
    for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
      read_image(&data_file, &label, str_buffer);
      datum.set_label(label);
      // copy raw image to padded image
      for(int c=0; c<3;c++){
    	  int src_offset = kCIFARSize*kCIFARSize*c;
    	  int dst_offset = padded_imange_size*padded_imange_size*c + pad*padded_imange_size+pad;
    	  for(int h=0; h<kCIFARSize;h++){
			  memcpy(str_buffer_pad+dst_offset,str_buffer+src_offset,kCIFARSize);
    		  src_offset += kCIFARSize;
    		  dst_offset += padded_imange_size;
    	  }
      }
      //datum.set_data(str_buffer, kCIFARImageNBytes);
      datum.set_data(str_buffer_pad, padded_imange_size*padded_imange_size*3);
      string out;
      CHECK(datum.SerializeToString(&out));
      txn->Put(caffe::format_int(fileid * kCIFARBatchSize + itemid, 5), out);
    }
  }
  txn->Commit();
  train_db->Close();

  LOG(INFO) << "Writing Testing data";
  scoped_ptr<db::DB> test_db(db::GetDB(db_type));
  test_db->Open(output_folder + "/cifar10_test_" + db_type, db::NEW);
  txn.reset(test_db->NewTransaction());
  // Open files
  std::ifstream data_file((input_folder + "/test_batch.bin").c_str(),
      std::ios::in | std::ios::binary);
  CHECK(data_file) << "Unable to open test file.";
  for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
    read_image(&data_file, &label, str_buffer);
    datum.set_label(label);
    // copy raw image to padded image
    for(int c=0; c<3;c++){
  	  int src_offset = kCIFARSize*kCIFARSize*c;
  	  int dst_offset = padded_imange_size*padded_imange_size*c + pad*padded_imange_size+pad;
	  for(int h=0; h<kCIFARSize;h++){
		  memcpy(str_buffer_pad+dst_offset,str_buffer+src_offset,kCIFARSize);
		  src_offset += kCIFARSize;
		  dst_offset += padded_imange_size;
	  }
    }
    //datum.set_data(str_buffer, kCIFARImageNBytes);
    datum.set_data(str_buffer_pad, padded_imange_size*padded_imange_size*3);
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(caffe::format_int(itemid, 5), out);
  }
  txn->Commit();
  test_db->Close();
  free(str_buffer_pad);
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = 1;
  if ((argc != 4) && (argc != 5)) {

    printf("This script converts the CIFAR dataset to the leveldb format used\n"
           "by caffe to perform classification.\n"
           "Usage:\n"
           "    convert_cifar_data input_folder output_folder db_type [pad]\n"
           "Where the input folder should contain the binary batch files.\n"
           "The CIFAR dataset could be downloaded at\n"
           "    http://www.cs.toronto.edu/~kriz/cifar.html\n"
           "You should gunzip them after downloading.\n");
  } else {
	int pad = 0;
	if(argc == 5) pad = atoi(argv[4]);
    google::InitGoogleLogging(argv[0]);
    convert_dataset(string(argv[1]), string(argv[2]), string(argv[3]), pad);
  }
  return 0;
}
