#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>
#include <parquet/properties.h>
#include <filesystem>
#include <iostream>
#include <map>
#include <random>

namespace fs = std::filesystem;
using arrow::Table;

std::mt19937 rng(42);  // fixed seed for reproducibility

template <typename Builder, typename ValueFunc>
std::shared_ptr<arrow::Array> generate_array(ValueFunc func, int64_t num_rows) {
  Builder builder;
  for (int64_t i = 0; i < num_rows; ++i) {
    builder.Append(func());
  }
  std::shared_ptr<arrow::Array> array;
  builder.Finish(&array);
  return array;
}

struct ColumnSpec {
  std::string encoding_name;
  parquet::Encoding::type encoding;
  std::string type_name;
  std::shared_ptr<arrow::DataType> arrow_type;
};

void write_parquet(const ColumnSpec& spec,
                   const std::shared_ptr<arrow::Array>& array) {
  auto field = arrow::field("data", spec.arrow_type);
  auto schema = arrow::schema({field});
  auto table = arrow::Table::Make(schema, {array});

  fs::create_directory("data/" + spec.encoding_name);
  std::string filename = "data/" + spec.encoding_name + "/" + spec.type_name + ".parquet";

  auto builder = parquet::WriterProperties::Builder();
  if (spec.encoding == parquet::Encoding::PLAIN_DICTIONARY ||
      spec.encoding == parquet::Encoding::RLE_DICTIONARY) {
    builder.enable_dictionary();
  } else if (spec.encoding == parquet::Encoding::DELTA_BINARY_PACKED ||
             spec.encoding == parquet::Encoding::DELTA_LENGTH_BYTE_ARRAY) {
    builder.disable_dictionary()->encoding("data", spec.encoding);
  } else {
    builder.encoding("data", spec.encoding);
  }
  auto props = builder.build();

  std::shared_ptr<arrow::io::OutputStream> outfile;
  PARQUET_ASSIGN_OR_THROW(outfile, arrow::io::FileOutputStream::Open(filename));
  PARQUET_THROW_NOT_OK(
      parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile, 1024, props));

  std::cout << "Wrote " << filename << "\n";
}

int main() {
  int64_t num_rows = 1000;

  // Generate base data once per type
  std::map<std::string, std::shared_ptr<arrow::Array>> base_data;
  base_data["int32"] = generate_array<arrow::Int32Builder>(
      [] { return std::uniform_int_distribution<int32_t>(0, 10000)(rng); }, num_rows);
  base_data["float"] = generate_array<arrow::FloatBuilder>(
      [] { return std::uniform_real_distribution<float>(0.f, 100.f)(rng); }, num_rows);
  base_data["int64"] = generate_array<arrow::Int64Builder>(
      [] { return static_cast<int64_t>(rng()) << 8; }, num_rows);
  base_data["string"] = generate_array<arrow::StringBuilder>(
      [] {
        int len = std::uniform_int_distribution<>(5, 10)(rng);
        std::string s(len, 'a' + (rng() % 26));
        return s;
      }, num_rows);
  base_data["binary"] = generate_array<arrow::BinaryBuilder>(
      [] {
        int len = std::uniform_int_distribution<>(3, 15)(rng);
        std::vector<uint8_t> bytes(len);
        for (int i = 0; i < len; ++i) bytes[i] = rng() % 256;
        return std::string(reinterpret_cast<const char*>(bytes.data()), len);
      }, num_rows);

  // Specs to write out
  std::vector<ColumnSpec> specs = {
      // PLAIN encoding
      {"PLAIN", parquet::Encoding::PLAIN, "string", arrow::utf8()},
      {"PLAIN", parquet::Encoding::PLAIN, "float", arrow::float32()},
      {"PLAIN", parquet::Encoding::PLAIN, "int32", arrow::int32()},
      {"PLAIN", parquet::Encoding::PLAIN, "binary", arrow::binary()},

      // PLAIN_DICTIONARY encoding
      {"PLAIN_DICTIONARY", parquet::Encoding::PLAIN_DICTIONARY, "string", arrow::utf8()},
      {"PLAIN_DICTIONARY", parquet::Encoding::PLAIN_DICTIONARY, "float", arrow::float32()},
      {"PLAIN_DICTIONARY", parquet::Encoding::PLAIN_DICTIONARY, "int32", arrow::int32()},
      {"PLAIN_DICTIONARY", parquet::Encoding::PLAIN_DICTIONARY, "int64", arrow::int64()},
      {"PLAIN_DICTIONARY", parquet::Encoding::PLAIN_DICTIONARY, "binary", arrow::binary()},

      {"RLE_DICTIONARY", parquet::Encoding::RLE_DICTIONARY, "string", arrow::utf8()},
      {"RLE_DICTIONARY", parquet::Encoding::RLE_DICTIONARY, "float", arrow::float32()},
      {"RLE_DICTIONARY", parquet::Encoding::RLE_DICTIONARY, "int32", arrow::int32()},
      {"RLE_DICTIONARY", parquet::Encoding::RLE_DICTIONARY, "int64", arrow::int64()},
      {"RLE_DICTIONARY", parquet::Encoding::RLE_DICTIONARY, "binary", arrow::binary()},

      {"RLE", parquet::Encoding::RLE, "string", arrow::utf8()},
      {"RLE", parquet::Encoding::RLE, "float", arrow::float32()},
      {"RLE", parquet::Encoding::RLE, "int32", arrow::int32()},
      {"RLE", parquet::Encoding::RLE, "int64", arrow::int64()},
      {"RLE", parquet::Encoding::RLE, "binary", arrow::binary()},


      // DELTA_BINARY_PACKED encoding (int types only)
      {"DELTA_BINARY_PACKED", parquet::Encoding::DELTA_BINARY_PACKED, "int32", arrow::int32()},
      {"DELTA_BINARY_PACKED", parquet::Encoding::DELTA_BINARY_PACKED, "int64", arrow::int64()},

      // DELTA_LENGTH_BYTE_ARRAY encoding (binary only)
      {"DELTA_LENGTH_BYTE_ARRAY", parquet::Encoding::DELTA_LENGTH_BYTE_ARRAY, "binary", arrow::binary()},
  };

  // Write files using pre-generated data per type
  for (const auto& spec : specs) {
    write_parquet(spec, base_data.at(spec.type_name));
  }

  return 0;
}
