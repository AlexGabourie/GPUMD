/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "at.cuh"
#include "utilities/error.cuh"

#define BLOCK_SIZE_AT 128

/*----------------------------------------------------------------------------80
This file implements the Axilrod-Teller potential.
[1] B.M. Axilrod and E. Teller,
    Interaction of the van der Waals Type Between Three Atoms,
    J. Chem. Phys. 11, 299 (1943).
    The implementation supports up to two atom types.
------------------------------------------------------------------------------*/

AT::AT(FILE* fid, int num_of_types, const Neighbor& neighbor)
{
  if (num_of_types == 1) {
    initialize_at_1(fid);
  }
  else {
    initialize_at_2(fid);
  }

  // memory for the partial forces dU_i/dr_ij
  const int num_of_neighbors = min(neighbor.MN, 50) * neighbor.NN.size();
  at_data.f12x.resize(num_of_neighbors);
  at_data.f12y.resize(num_of_neighbors);
  at_data.f12z.resize(num_of_neighbors);
}

void AT::initialize_at_1(FILE* fid)
{
  printf("Use single-element Axilrod-Teller potential.\n");
  int count, rcnum;
  double z, rcglobal, rc3;
  count = fscanf(fid, "%lf%lf%lf%d", &z, &rcglobal, &rc3, &rcnum);
  PRINT_SCANF_ERROR(count, 4, "Reading error for AT potential.");

  at_para.z[0] = z;
  if (rcglobal <= 0.0 || rc3 <= 0.0){
    PRINT_INPUT_ERROR("AT potential error: Cutoffs must be positive.\n");
  }
    rc = rcglobal;
  at_para.rc2 = rc*rc;
  at_para.rc6 = rc3*rc3;
  if (rcnum != 2 && rcnum != 3){
    PRINT_INPUT_ERROR("AT potential error: Number of enforcable cutoffs should be 2 or 3.\n");
  }
  at_para.rcnum = rcnum;
}

void AT::initialize_at_2(FILE* fid)
{
  printf("Use two-element Axilrod-Teller potential.\n");
  int count, rcnum;
  double z[4], rcglobal, rc3;
  count = fscanf(fid, "%lf%lf%lf%lf%lf%lf%d", &z[0], &z[1], &z[2], &z[3],&rcglobal, &rc3, &rcnum);
  PRINT_SCANF_ERROR(count, 7, "Reading error for AT potential.");

  at_para.z[0] = z[0];
  at_para.z[7] = z[1];
  at_para.z[1] = at_para.z[2] = at_para.z[4] = z[2];
  at_para.z[3] = at_para.z[5] = at_para.z[6] = z[3];
  if (rcglobal <= 0.0 || rc3 <= 0.0){
      PRINT_INPUT_ERROR("AT potential error: Cutoffs must be positive.\n");
    }
  rc = rcglobal;
  at_para.rc2 = rc*rc;
  at_para.rc6 = rc3*rc3;
  if (rcnum != 2 && rcnum != 3){
      PRINT_INPUT_ERROR("AT potential error: Number of enforcable cutoffs should be 2 or 3.\n");
    }
  at_para.rcnum = rcnum;
}

AT::~AT(void)
{
  // nothing
}

static __global__ void gpu_set_f12_to_zero(
  const int N,
  const int N1,
  const int N2,
  const int* g_NN,
  double* g_f12x,
  double* g_f12y,
  double* g_f12z)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; // particle index
  if (n1 >= N1 && n1 < N2) {
    int neighbor_number = g_NN[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      g_f12x[index] = 0.0;
      g_f12y[index] = 0.0;
      g_f12z[index] = 0.0;
    }
  }
}


static __global__ void gpu_find_force_at_partial(
  const int number_of_atoms,
  const int N1,
  const int N2,
  const Box box,
  const AT_Para at,
  const int* g_neighbor_number,
  const int* g_neighbor_list,
  const int* g_type,
  const int shift,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_potential,
  double* g_f12x,
  double* g_f12y,
  double* g_f12z)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N1 && n1 < N2){
    int neighbor_number = g_neighbor_number[n1];
    int type1 = g_type[n1] - shift;
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    double potential_energy = 0.0;

    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * number_of_atoms + n1;
      int n2 = g_neighbor_list[index];
      int type2 = g_type[n2] - shift;
      int tally_12 = 1; // denotes valid pair
      double x2 = g_x[n2];
      double y2 = g_y[n2];
      double z2 = g_z[n2];
      double x12 = x2 - x1;
      double y12 = y2 - y1;
      double z12 = z2 - z1;
      apply_mic(box, x12, y12, z12);
      double d12d12 = x12*x12 + y12*y12 + z12*z12;
      if (d12d12 > at.rc2) {
        tally_12 = 0;
      }
      double d12d12inv = 1/d12d12;
      double f12x, f12y, f12z;
      f12x = f12y = f12z = 0;

      for (int i2 = 0; i2 < neighbor_number; ++i2) {
        int n3 = g_neighbor_list[n1 + number_of_atoms * i2];
        if (n3 == n2) {
          continue;
        }
        int type3 = g_type[n3] - shift;
        int tally_13 = 1;
        int tally_23 = 1;
        double x3 = g_x[n3];
        double y3 = g_y[n3];
        double z3 = g_z[n3];
        double x13 = x3 - x1;
        double y13 = y3 - y1;
        double z13 = z3 - z1;
        apply_mic(box, x13, y13, z13);
        double d13d13 = x13*x13 + y13*y13 + z13*z13;
        if (d13d13 > at.rc2) {
          tally_13 = 0;
        }
        double d13d13inv = 1/d13d13;

        double x23 = x3 - x2;
        double y23 = y3 - y2;
        double z23 = z3 - z2;
        apply_mic(box, x23, y23, z23);
        double d23d23 = x23*x23 + y23*y23 + z23*z23;
        if (d23d23 > at.rc2) {
          tally_23 = 0;
        }
        double dist2_prod = d12d12*d13d13*d23d23;
        if (dist2_prod > at.rc6){
          continue;
        }

        if (tally_12 + tally_13 + tally_23 < at.rcnum){
          continue;
        }

        double z = at.z[(type1<<2)+(type2<<1)+type3];
        double scale = z/(dist2_prod*dist2_prod*sqrt(dist2_prod));
        double d12d13 = x12*x13 + y12*y13 + z12*z13;
        double d12d23 = x12*x23 + y12*y23 + z12*z23;
        double d13d23 = x13*x23 + y13*y23 + z13*z23;
        double ddd = d12d13*d12d23*d13d23;

        potential_energy += scale*dist2_prod - 3.0*scale*ddd;

        double tmp1 = 5*ddd*d12d12inv - d12d23*d13d23 - d13d13*d23d23;
        double tmp2 = 5*ddd*d13d13inv - d12d23*d13d23 - d12d12*d23d23;
        double tmp3 = -1*(d12d13*d13d23 + d12d13*d12d23);

        f12x += scale*(tmp1*x12 + tmp2*x13 + tmp3*x23);
        f12y += scale*(tmp1*y12 + tmp2*y13 + tmp3*y23);
        f12z += scale*(tmp1*z12 + tmp2*z13 + tmp3*z23);
      }
      g_f12x[index] = f12x;
      g_f12y[index] = f12y;
      g_f12z[index] = f12z;

    }
    // save potential
    g_potential[n1] = potential_energy/6.0;

  }
}

void AT::compute(
  const int type_shift,
  const Box& box,
  const Neighbor& neighbor,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int number_of_atoms = type.size();
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_AT + 1;

  gpu_set_f12_to_zero<<<grid_size, BLOCK_SIZE_AT>>>(
    number_of_atoms, N1, N2, neighbor.NN_local.data(), at_data.f12x.data(), at_data.f12y.data(),
    at_data.f12z.data());
  CUDA_CHECK_KERNEL

  // step 1: calculate the partial forces
  gpu_find_force_at_partial<<<grid_size, BLOCK_SIZE_AT>>>(
    number_of_atoms, N1, N2, box, at_para, neighbor.NN_local.data(), neighbor.NL_local.data(),
    type.data(), type_shift, position_per_atom.data(), position_per_atom.data() + number_of_atoms,
    position_per_atom.data() + number_of_atoms * 2, potential_per_atom.data(), at_data.f12x.data(),
    at_data.f12y.data(), at_data.f12z.data());
  CUDA_CHECK_KERNEL

  // step 2: calculate force and related quantities
  find_properties_many_body(
    box, neighbor.NN_local.data(), neighbor.NL_local.data(), at_data.f12x.data(),
    at_data.f12y.data(), at_data.f12z.data(), position_per_atom, force_per_atom, virial_per_atom);
}
