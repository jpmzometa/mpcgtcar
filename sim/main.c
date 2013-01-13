#include "stdio.h"
#include "cmpc/include/mpc.h"

int main(void)
{
	int i, j;
	const real_t X_SP[MPC_STATES] = {1.0, 0.};  /* linearization point */
	const real_t U_OS[MPC_INPUTS] = {0., -0.2464};  /* offset in inputs */
	real_t x[MPC_STATES];  /* state used in the MPC algorithm */
	real_t x_sys[MPC_STATES];  /* current state of the system */
	real_t u_sys[MPC_INPUTS];  /* input to the system */

	extern struct mpc_ctl ctl;  /* already defined */

	ctl.conf->in_iter = 100;  /* number of iterations */
	ctl.conf->warmstart = 1;

	/* The initial state */
	x[0] = 0.;
	x[1] = 0.;

	for (i=0; i<100; i++) {
		/* Solve MPC problem */
		for (j=0; j<MPC_STATES; j++) {
			x[j] = x_sys[j] - X_SP[j];
		} /* MPC regulates the origin */
		mpc_ctl_solve_problem(&ctl, x);  /* solve the MPC problem */

		for (j=0; j<MPC_INPUTS; j++) {
			u_sys[j] = ctl.u_opt[j] + U_OS[j];
		}  /* the real car has offset in inputs */

		printf("u[0] = %f; u[1] = %f \n", u_sys[0], u_sys[1]);
	}
	printf("\n SUCCESS! \n");

	return 0;
}

