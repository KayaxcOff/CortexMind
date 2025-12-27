#include "CortexMind/core/Engine/Tensor/tensor.hpp"

using namespace cortex::_fw;

std::vector<int64_t> MindTensor::compute_strides(const std::vector<int64_t>& shape) noexcept {
	std::vector<int64_t> strides(shape.size());
	int64_t stride = 1;
	for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
		strides[i] = stride;
		stride *= shape[i];
	}
	return strides;
}

int64_t MindTensor::compute_offset(const std::vector<int64_t>& indices) const noexcept {
	int64_t offset = 0;
	for (size_t i = 0; i < indices.size(); ++i) {
		err::IsStatusFailed(indices[i] >= 0 && indices[i] < m_shape[i], "cortex::_fw::MindTensor::compute_offset()", "Out-of-bounds access detected.");
		offset += indices[i] * this->m_strides[i];
	}
	return offset + this->m_offset;
}

bool MindTensor::is_contiguous() const noexcept {
	return this->m_strides == compute_strides(this->m_shape);
}

MindTensor::MindTensor(bool requires_grad) : m_shape({}), m_strides({}), m_offset(0), m_stor(std::make_shared<TensorStorage>(0)) {
	if (requires_grad) {
		this->m_meta = std::make_shared<meta::AutogradMeta>();
		this->m_meta->requires_grad = true;
		this->m_meta->is_leaf = true;
	}
}

MindTensor::MindTensor(std::vector<int64_t> shape, bool requires_grad) : m_shape(std::move(shape)), m_strides(compute_strides(this->m_shape)), m_offset(0) {
	size_t total = this->m_strides.empty() ? 0 : (this->m_strides[0] * this->m_shape[0]);

	this->m_stor = std::make_shared<TensorStorage>(total);

	if (requires_grad) {
		this->m_meta = std::make_shared<meta::AutogradMeta>();
		this->m_meta->requires_grad = true;
		this->m_meta->is_leaf = true;
	}
}

MindTensor::MindTensor(std::initializer_list<int64_t> shape, bool requires_grad) : MindTensor(std::vector<int64_t>(shape), requires_grad) {}

MindTensor::MindTensor(const std::vector<int64_t>& shape, const float* data, bool requires_grad) : MindTensor(shape, requires_grad) {
	size_t total = 1;
	for (const auto& dim : this->m_shape) {
		total *= dim;
	}
	err::IsStatusFailed(data != nullptr, "cortex::_fw::MindTensor::MindTensor()", "Input data pointer is null.");
	std::memcpy(this->m_stor->ptr(), data, total * sizeof(float));
}

float& MindTensor::at(const std::vector<int64_t>& indices) noexcept {
	int64_t offset = compute_offset(indices);
	return this->m_stor->ptr()[offset];
}

const float& MindTensor::at(const std::vector<int64_t>& indices) const noexcept {
	int64_t offset = compute_offset(indices);
	return this->m_stor->ptr()[offset];
}

float* MindTensor::data() noexcept {
	return this->m_stor->ptr() + this->m_offset;
}

const float* MindTensor::data() const noexcept {
	return this->m_stor->ptr() + this->m_offset;
}

const std::vector<int64_t>& MindTensor::shape() const noexcept {
	return this->m_shape;
}

const std::vector<int64_t>& MindTensor::strides() const noexcept {
	return this->m_strides;
}

size_t MindTensor::numel() const noexcept {
	size_t total = 1;
	for (const auto& dim : this->m_shape) {
		total *= dim;
	}
	return total;
}

std::size_t MindTensor::size() const noexcept {
	return this->m_stor->size();
}

bool MindTensor::empty() const noexcept {
	return this->m_stor->empty();
}

bool MindTensor::requires_grad() const noexcept {
	return this->m_meta && this->m_meta->requires_grad;
}

void MindTensor::backward() noexcept {
	// Backward fonksiyonu burada uygulanacak
}

void MindTensor::print() const noexcept {
	if (m_shape.empty()) {
		std::cout << "[]\n";
		return;
	}

	const int indent_step = 2;

	std::function<void(int64_t, int64_t, int)> PrintRecursive;
	PrintRecursive = [&](int64_t dim, int64_t offset, int indent) {
		if (dim == static_cast<int64_t>(m_shape.size()) - 1) {
			std::cout << std::string(indent, ' ') << "[";
			for (int64_t i = 0; i < m_shape[dim]; ++i) {
				std::cout << std::fixed << std::setprecision(4)
					<< this->m_stor->ptr()[offset + i * this->m_strides[dim]];
				if (i != this->m_shape[dim] - 1) std::cout << ", ";
			}
			std::cout << "]";
		}
		else {
			std::cout << std::string(indent, ' ') << "[\n";
			for (int64_t i = 0; i < this->m_shape[dim]; ++i) {
				PrintRecursive(dim + 1, offset + i * this->m_strides[dim], indent + indent_step);
				if (i != this->m_shape[dim] - 1) std::cout << ",\n";
				else std::cout << "\n";
			}
			std::cout << std::string(indent, ' ') << "]";
		}
		};

	PrintRecursive(0, this->m_offset, 0);
	std::cout << std::endl;
}

void MindTensor::uniform_rand(float lower, float upper) noexcept {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(lower, upper);

	size_t total = numel();
	float* ptr = this->m_stor->ptr() + this->m_offset;

	for (size_t i = 0; i < total; ++i) {
		ptr[i] = dis(gen);
	}
}

void MindTensor::zero() noexcept {
	size_t total = numel();
	float* ptr = this->m_stor->ptr() + this->m_offset;
	std::size_t i = 0;

	avx2::vec8f vz = avx2::zero();

	for (; i + 8 <= total; i += 8) {
		avx2::storeu(ptr + i, vz);
	}
	for (; i < total; ++i) {
		ptr[i] = 0.0f;
	}
}

void MindTensor::ones() noexcept {
	size_t total = numel();
	float* ptr = this->m_stor->ptr() + this->m_offset;
	std::size_t i = 0;

	avx2::vec8f vo = avx2::broadcast(1.0f);

	for (; i + 8 <= total; i += 8) {
		avx2::storeu(ptr + i, vo);
	}
	for (; i < total; ++i) {
		ptr[i] = 1.0f;
	}
}

void MindTensor::fill(float value) noexcept {
	size_t total = numel();
	float* ptr = this->m_stor->ptr() + this->m_offset;
	std::size_t i = 0;

	avx2::vec8f vval = avx2::broadcast(value);

	for (; i + 8 <= total; i += 8) {
		avx2::storeu(ptr + i, vval);
	}
	for (; i < total; ++i) {
		ptr[i] = value;
	}
}

void MindTensor::allocate() noexcept {
	if (!this->m_stor || this->m_stor->size() == 0) {
		size_t total = numel();
		this->m_stor = std::make_shared<TensorStorage>(total);
		this->m_offset = 0;
	}
}

void MindTensor::resize(const std::vector<int64_t>& new_shape) noexcept {
	this->m_shape = new_shape;
	this->m_strides = compute_strides(this->m_shape);

	size_t new_total = 1;
	for (auto dim : this->m_shape) new_total *= dim;

	this->m_stor = std::make_shared<TensorStorage>(new_total);
	this->m_offset = 0;
}

void MindTensor::require_grad(bool requires_grad) noexcept {
	if (requires_grad) {
		if (!this->m_meta) {
			this->m_meta = std::make_shared<meta::AutogradMeta>();
		}
		this->m_meta->requires_grad = true;
	}
	else {
		if (this->m_meta) {
			this->m_meta->requires_grad = false;
		}
	}
}

MindTensor MindTensor::flatten() const noexcept {
	std::vector<int64_t> new_shape = { static_cast<int64_t>(numel()) };
	MindTensor output(new_shape, this->requires_grad());
	
	output.m_stor = this->m_stor;
	output.m_offset = this->m_offset;

	return output;
}

MindTensor MindTensor::matmul(const MindTensor& other) const noexcept {
	err::IsStatusFailed(this->m_shape.size() == 2 && other.m_shape.size() == 2, "cortex::_fw::MindTensor::matmul()", "Both tensors must be 2D for matrix multiplication.");
	err::IsStatusFailed(this->m_shape[1] == other.m_shape[0], "cortex::_fw::MindTensor::matmul()", "Inner dimensions must match for matrix multiplication.");
	
	std::vector<int64_t> out_shape = { this->shape()[0], other.shape()[1] };
	MindTensor output(out_shape, this->requires_grad() || other.requires_grad());

	avx2::matrix_t::matmul(
		this->data() + this->m_offset,
		other.data() + other.m_offset,
		output.data() + output.m_offset,
		static_cast<size_t>(this->m_shape[0]),
		static_cast<size_t>(this->m_shape[1]),
		static_cast<size_t>(other.m_shape[1])
	);

	return output;
}

MindTensor MindTensor::transpose() const noexcept {
	err::IsStatusFailed(this->m_shape.size() == 2, "cortex::_fw::MindTensor::transpose()", "Tensor must be 2D to transpose.");

	MindTensor output(this->m_shape, this->requires_grad());

	std::swap(output.m_shape[0], output.m_shape[1]);
	std::swap(output.m_strides[0], output.m_strides[1]);

	output.m_offset = this->m_offset;
	output.m_stor = this->m_stor;

	return output;
}

MindTensor MindTensor::permute(std::vector<int64_t> axes) const noexcept {
	err::IsStatusFailed(axes.size() == this->m_shape.size(), "cortex::_fw::MindTensor::permute()", "Number of axes must match tensor dimensions.");
	MindTensor output(this->m_shape, this->requires_grad());
	for (size_t i = 0; i < axes.size(); ++i) {
		err::IsStatusFailed(axes[i] >= 0 && axes[i] < static_cast<int64_t>(this->m_shape.size()), "cortex::_fw::MindTensor::permute()", "Invalid axis index.");
		output.m_shape[i] = this->m_shape[axes[i]];
		output.m_strides[i] = this->m_strides[axes[i]];
	}
	output.m_offset = this->m_offset;
	output.m_stor = this->m_stor;
	return output;
}

MindTensor MindTensor::slice(int64_t dim, int64_t start, int64_t end) const noexcept {
	err::IsStatusFailed(dim >= 0 && dim < static_cast<int64_t>(this->m_shape.size()), "cortex::_fw::MindTensor::slice()", "Dimension out of range.");
	err::IsStatusFailed(start >= 0 && end <= this->m_shape[dim] && start < end, "cortex::_fw::MindTensor::slice()", "Invalid slice indices.");
	MindTensor output(this->m_shape, this->requires_grad());
	output.m_shape[dim] = end - start;
	output.m_offset = this->m_offset + start * this->m_strides[dim];
	output.m_stor = this->m_stor;
	return output;
}

MindTensor MindTensor::copy() const noexcept {
	MindTensor output(this->m_shape, this->requires_grad());
	size_t total = numel();
	std::memcpy(output.m_stor->ptr(), this->m_stor->ptr() + this->m_offset, total * sizeof(float));
	return output;
}

MindTensor MindTensor::sum() const noexcept {
	// Tek bir skalar döneceðimiz için shape boþ
	MindTensor output({}, this->requires_grad());

	const float* ptr = this->m_stor->ptr() + this->m_offset;
	size_t n = numel();

	avx2::vec8f vtotal = avx2::zero();

	size_t i = 0;
	for (; i + 8 <= n; i += 8) {
		avx2::vec8f vdata = avx2::loadu(ptr + i);
		vtotal = avx2::add(vtotal, vdata);
	}

	float tail_total = 0.0f;
	for (; i < n; ++i) tail_total += ptr[i];

	avx2::vec4f hsum = avx2::horizontal_add(vtotal);

	float total = hsum[0] + hsum[1] + hsum[2] + hsum[3] + tail_total;

	output.m_stor->ptr()[0] = total;

	return output;
}

MindTensor MindTensor::sum(int64_t dim, bool keepdim) const noexcept {
	err::IsStatusFailed(dim >= 0 && dim < static_cast<int64_t>(m_shape.size()), "cortex::_fw::MindTensor::sum()", "Dimension out of range.");

	std::vector<int64_t> out_shape;
	for (size_t i = 0; i < m_shape.size(); ++i) {
		if (i == static_cast<size_t>(dim)) {
			if (keepdim) out_shape.push_back(1);
		}
		else {
			out_shape.push_back(m_shape[i]);
		}
	}

	MindTensor output(out_shape, this->requires_grad());
	output.zero();

	const float* src = this->data();
	float* dst = output.data();

	std::function<void(size_t, size_t, size_t)> sum_recursive;
	sum_recursive = [&](size_t dim_idx, size_t src_offset, size_t dst_offset) {
		if (dim_idx == m_shape.size()) {
			dst[dst_offset] += src[src_offset];
			return;
		}

		if (dim_idx == static_cast<size_t>(dim)) {
			for (int64_t i = 0; i < m_shape[dim_idx]; ++i) {
				sum_recursive(dim_idx + 1,
					src_offset + i * m_strides[dim_idx],
					dst_offset);
			}
		}
		else {
			size_t dst_dim_idx = dim_idx - (dim_idx > static_cast<size_t>(dim) && !keepdim ? 1 : 0);
			for (int64_t i = 0; i < m_shape[dim_idx]; ++i) {
				sum_recursive(dim_idx + 1,
					src_offset + i * m_strides[dim_idx],
					dst_offset + i * output.m_strides[dst_dim_idx]);
			}
		}
		};

	sum_recursive(0, 0, 0);

	return output;
}

MindTensor MindTensor::mean() const noexcept {
	MindTensor output = this->sum();

	float divisor = static_cast<float>(this->numel());

	float* ptr = output.m_stor->ptr() + output.m_offset;
	size_t total = output.numel();
	avx2::vec8f vdiv = avx2::broadcast(divisor);

	size_t i = 0;
	for (; i + 8 <= total; i += 8) {
		avx2::vec8f vdata = avx2::loadu(ptr + i);
		avx2::storeu(ptr + i, avx2::div(vdata, vdiv));
	}
	for (; i < total; ++i) {
		ptr[i] /= divisor;
	}

	return output;
}

MindTensor MindTensor::mean(int64_t dim, bool keepdim) const noexcept {
	MindTensor sum_tensor = this->sum(dim, keepdim);
	float divisor = static_cast<float>(this->m_shape[dim]);
	size_t total = sum_tensor.numel();
	float* ptr = sum_tensor.m_stor->ptr() + sum_tensor.m_offset;
	std::size_t i = 0;
	avx2::vec8f vdiv = avx2::broadcast(divisor);
	for (; i + 8 <= total; i += 8) {
		avx2::vec8f vdata = avx2::loadu(ptr + i);
		avx2::vec8f vresult = avx2::div(vdata, vdiv);
		avx2::storeu(ptr + i, vresult);
	}
	for (; i < total; ++i) {
		ptr[i] /= divisor;
	}
	return sum_tensor;
}

MindTensor MindTensor::view(const std::vector<int64_t>& new_shape) const {
	size_t new_numel = 1;
	for (const auto& dim : new_shape) {
		new_numel *= dim;
	}
	err::IsStatusFailed(new_numel == this->numel(), "cortex::_fw::MindTensor::view()", "Total number of elements must remain the same in view.");
	MindTensor output(new_shape, this->requires_grad());
	output.m_stor = this->m_stor;
	output.m_offset = this->m_offset;
	output.m_strides = compute_strides(new_shape);
	return output;
}

MindTensor MindTensor::operator+(const MindTensor& other) const noexcept {
	err::IsStatusFailed(this->m_shape == other.m_shape, "cortex::_fw::MindTensor::operator+()", "Shapes must match for element-wise addition.");
	MindTensor output(this->m_shape, this->requires_grad() || other.requires_grad());
	size_t total = numel();
	const float* ptr1 = this->m_stor->ptr() + this->m_offset;
	const float* ptr2 = other.m_stor->ptr() + other.m_offset;
	float* out_ptr = output.m_stor->ptr() + output.m_offset;
	avx2::matrix_t::add(ptr1, ptr2, out_ptr, total);

	return output;
}

MindTensor MindTensor::operator-(const MindTensor& other) const noexcept {
	err::IsStatusFailed(this->m_shape == other.m_shape, "cortex::_fw::MindTensor::operator-()", "Shapes must match for element-wise subtraction.");
	MindTensor output(this->m_shape, this->requires_grad() || other.requires_grad());
	size_t total = numel();
	const float* ptr1 = this->m_stor->ptr() + this->m_offset;
	const float* ptr2 = other.m_stor->ptr() + other.m_offset;
	float* out_ptr = output.m_stor->ptr() + output.m_offset;
	avx2::matrix_t::sub(ptr1, ptr2, out_ptr, total);
	return output;
}

MindTensor MindTensor::operator*(const MindTensor& other) const noexcept {
	err::IsStatusFailed(this->m_shape == other.m_shape, "cortex::_fw::MindTensor::operator*()", "Shapes must match for element-wise multiplication.");
	MindTensor output(this->m_shape, this->requires_grad() || other.requires_grad());
	size_t total = numel();
	const float* ptr1 = this->m_stor->ptr() + this->m_offset;
	const float* ptr2 = other.m_stor->ptr() + other.m_offset;
	float* out_ptr = output.m_stor->ptr() + output.m_offset;
	avx2::matrix_t::mul(ptr1, ptr2, out_ptr, total);
	return output;
}

MindTensor MindTensor::operator/(const MindTensor& other) const noexcept {
	err::IsStatusFailed(this->m_shape == other.m_shape, "cortex::_fw::MindTensor::operator/()", "Shapes must match for element-wise division.");
	MindTensor output(this->m_shape, this->requires_grad() || other.requires_grad());
	size_t total = numel();
	const float* ptr1 = this->m_stor->ptr() + this->m_offset;
	const float* ptr2 = other.m_stor->ptr() + other.m_offset;
	float* out_ptr = output.m_stor->ptr() + output.m_offset;
	avx2::matrix_t::div(ptr1, ptr2, out_ptr, total);
	return output;
}

MindTensor& MindTensor::operator+=(const MindTensor& other) noexcept {
	err::IsStatusFailed(this->m_shape == other.m_shape, "cortex::_fw::MindTensor::operator+=()", "Shapes must match for element-wise addition.");
	size_t total = numel();
	float* ptr1 = this->m_stor->ptr() + this->m_offset;
	const float* ptr2 = other.m_stor->ptr() + other.m_offset;
	avx2::matrix_t::add(ptr1, ptr2, ptr1, total);
	return *this;
}

MindTensor& MindTensor::operator-=(const MindTensor& other) noexcept {
	err::IsStatusFailed(this->m_shape == other.m_shape, "cortex::_fw::MindTensor::operator-=()", "Shapes must match for element-wise subtraction.");
	size_t total = numel();
	float* ptr1 = this->m_stor->ptr() + this->m_offset;
	const float* ptr2 = other.m_stor->ptr() + other.m_offset;
	avx2::matrix_t::sub(ptr1, ptr2, ptr1, total);
	return *this;
}

MindTensor& MindTensor::operator*=(const MindTensor& other) noexcept {
	err::IsStatusFailed(this->m_shape == other.m_shape, "cortex::_fw::MindTensor::operator*=()", "Shapes must match for element-wise multiplication.");
	size_t total = numel();
	float* ptr1 = this->m_stor->ptr() + this->m_offset;
	const float* ptr2 = other.m_stor->ptr() + other.m_offset;
	avx2::matrix_t::mul(ptr1, ptr2, ptr1, total);
	return *this;
}

MindTensor& MindTensor::operator/=(const MindTensor& other) noexcept {
	err::IsStatusFailed(this->m_shape == other.m_shape, "cortex::_fw::MindTensor::operator/=()", "Shapes must match for element-wise division.");
	size_t total = numel();
	float* ptr1 = this->m_stor->ptr() + this->m_offset;
	const float* ptr2 = other.m_stor->ptr() + other.m_offset;
	avx2::matrix_t::div(ptr1, ptr2, ptr1, total);
	return *this;
}

MindTensor MindTensor::operator+(float scalar) const noexcept {
	MindTensor output(this->m_shape, this->requires_grad());
	size_t total = numel();
	const float* ptr1 = this->m_stor->ptr() + this->m_offset;
	float* out_ptr = output.m_stor->ptr() + output.m_offset;
	avx2::vec8f vscalar = avx2::broadcast(scalar);
	size_t i = 0;
	for (; i + 8 <= total; i += 8) {
		avx2::vec8f vdata = avx2::loadu(ptr1 + i);
		avx2::vec8f vresult = avx2::add(vdata, vscalar);
		avx2::storeu(out_ptr + i, vresult);
	}
	for (; i < total; ++i) {
		out_ptr[i] = ptr1[i] + scalar;
	}
	return output;
}

MindTensor MindTensor::operator-(float scalar) const noexcept {
	MindTensor output(this->m_shape, this->requires_grad());
	size_t total = numel();
	const float* ptr1 = this->m_stor->ptr() + this->m_offset;
	float* out_ptr = output.m_stor->ptr() + output.m_offset;
	avx2::vec8f vscalar = avx2::broadcast(scalar);
	size_t i = 0;
	for (; i + 8 <= total; i += 8) {
		avx2::vec8f vdata = avx2::loadu(ptr1 + i);
		avx2::vec8f vresult = avx2::sub(vdata, vscalar);
		avx2::storeu(out_ptr + i, vresult);
	}
	for (; i < total; ++i) {
		out_ptr[i] = ptr1[i] - scalar;
	}
	return output;
}

MindTensor MindTensor::operator*(float scalar) const noexcept {
	MindTensor output(this->m_shape, this->requires_grad());
	size_t total = numel();
	const float* ptr1 = this->m_stor->ptr() + this->m_offset;
	float* out_ptr = output.m_stor->ptr() + output.m_offset;
	avx2::vec8f vscalar = avx2::broadcast(scalar);
	size_t i = 0;
	for (; i + 8 <= total; i += 8) {
		avx2::vec8f vdata = avx2::loadu(ptr1 + i);
		avx2::vec8f vresult = avx2::mul(vdata, vscalar);
		avx2::storeu(out_ptr + i, vresult);
	}
	for (; i < total; ++i) {
		out_ptr[i] = ptr1[i] * scalar;
	}
	return output;
}

MindTensor MindTensor::operator/(float scalar) const noexcept {
	MindTensor output(this->m_shape, this->requires_grad());
	size_t total = numel();
	const float* ptr1 = this->m_stor->ptr() + this->m_offset;
	float* out_ptr = output.m_stor->ptr() + output.m_offset;
	avx2::vec8f vscalar = avx2::broadcast(scalar);
	size_t i = 0;
	for (; i + 8 <= total; i += 8) {
		avx2::vec8f vdata = avx2::loadu(ptr1 + i);
		avx2::vec8f vresult = avx2::div(vdata, vscalar);
		avx2::storeu(out_ptr + i, vresult);
	}
	for (; i < total; ++i) {
		out_ptr[i] = ptr1[i] / scalar;
	}
	return output;
}

MindTensor& MindTensor::operator+=(float scalar) noexcept {
	size_t total = numel();
	float* ptr1 = this->m_stor->ptr() + this->m_offset;
	avx2::vec8f vscalar = avx2::broadcast(scalar);
	size_t i = 0;
	for (; i + 8 <= total; i += 8) {
		avx2::vec8f vdata = avx2::loadu(ptr1 + i);
		avx2::vec8f vresult = avx2::add(vdata, vscalar);
		avx2::storeu(ptr1 + i, vresult);
	}
	for (; i < total; ++i) {
		ptr1[i] += scalar;
	}
	return *this;
}

MindTensor& MindTensor::operator-=(float scalar) noexcept {
	size_t total = numel();
	float* ptr1 = this->m_stor->ptr() + this->m_offset;
	avx2::vec8f vscalar = avx2::broadcast(scalar);
	size_t i = 0;
	for (; i + 8 <= total; i += 8) {
		avx2::vec8f vdata = avx2::loadu(ptr1 + i);
		avx2::vec8f vresult = avx2::sub(vdata, vscalar);
		avx2::storeu(ptr1 + i, vresult);
	}
	for (; i < total; ++i) {
		ptr1[i] -= scalar;
	}
	return *this;
}

MindTensor& MindTensor::operator*=(float scalar) noexcept {
	size_t total = numel();
	float* ptr1 = this->m_stor->ptr() + this->m_offset;
	avx2::vec8f vscalar = avx2::broadcast(scalar);
	size_t i = 0;
	for (; i + 8 <= total; i += 8) {
		avx2::vec8f vdata = avx2::loadu(ptr1 + i);
		avx2::vec8f vresult = avx2::mul(vdata, vscalar);
		avx2::storeu(ptr1 + i, vresult);
	}
	for (; i < total; ++i) {
		ptr1[i] *= scalar;
	}
	return *this;
}

MindTensor& MindTensor::operator/=(float scalar) noexcept {
	size_t total = numel();
	float* ptr1 = this->m_stor->ptr() + this->m_offset;
	avx2::vec8f vscalar = avx2::broadcast(scalar);
	size_t i = 0;
	for (; i + 8 <= total; i += 8) {
		avx2::vec8f vdata = avx2::loadu(ptr1 + i);
		avx2::vec8f vresult = avx2::div(vdata, vscalar);
		avx2::storeu(ptr1 + i, vresult);
	}
	for (; i < total; ++i) {
		ptr1[i] /= scalar;
	}
	return *this;
}