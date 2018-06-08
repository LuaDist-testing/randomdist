---------------------------------------------------------------------
--     This Lua5 module is Copyright (c) 2017, Peter J Billam      --
--                       www.pjb.com.au                            --
--  This module is free software; you can redistribute it and/or   --
--         modify it under the same terms as Lua5 itself.          --
---------------------------------------------------------------------
-- Example usage:
-- local MM = require 'mymodule'
-- MM.foo()

local M = {} -- public interface
M.Version = '1.0'
M.VersionDate = '5jul2017'

------------------------------ private ------------------------------
function warn(...)
    local a = {}
    for k,v in pairs{...} do table.insert(a, tostring(v)) end
    io.stderr:write(table.concat(a),' ') ; io.stderr:flush()
end
function die(...) warn(...);  os.exit(1) end
function qw(s)  -- t = qw[[ foo  bar  baz ]]
    local t = {} ; for x in s:gmatch("%S+") do t[#t+1] = x end ; return t
end


------------------------------ public ------------------------------
-- http://www.design.caltech.edu/erik/Misc/Gaussian.html
-- The polar form of the Box-Muller transformation is faster and more robust:
--   float x1, x2, w, y1, y2;
--   do {
--      x1 = 2.0 * ranf() - 1.0;
--      x2 = 2.0 * ranf() - 1.0;
--      w = x1 * x1 + x2 * x2;
--   } while ( w >= 1.0 );
--   w = sqrt( (-2.0 * log( w ) ) / w );
--   y1 = x1 * w;
--   y2 = x2 * w;
-- where ranf() obtains a random number uniformly distributed in [0,1]
function M.new_grand (mean,stddev)
	local already = false
	local x1, x2, y1, y2
	return function (arg)
		if arg == 'reset' then already = false ; return nil end
		if already then already = false ; return mean + stddev*y2  end
		local w
		while true do
			x1 = 2.0*math.random() - 1.0
			x2 = 2.0*math.random() - 1.0
			w  = x1*x1 + x2*x2
			if w <= 1.0 then  break end
		end
		w  = math.sqrt( -2.0*math.log(w) / w )
		y1 = x1 * w
		y2 = x2 * w
		already = true
		return mean + stddev*y1
	end
end
--function M.grand (mean,stddev)
--	return mean + stddev*gauss_rand()
--end

function M.new_gue_irand (av)
    -- from av, we put together a sufficient array of probabilites
    -- of the various integers around av
	local pi  = math.pi
	local sin = math.sin
	local e   = 2.718281828
    local cumul = {}
    cumul[0] = 0
    for is = 1, math.floor(4*av + 0.5) do
        local s = is / av
        cumul[is] = cumul[is-1] + (32 / pi^2) * s^2 * e^((-4/pi) * s^2) / av
    end
    return function ()
        -- use math.random() to choose one of those integers ...
        local ran = math.random()   -- 0..1
        for i in ipairs(cumul) do
            if cumul[i] > ran then return i end
        end
        return #cumul   -- just in case ran is extremely close to 1.0
    end
end

function M.randomget(a)
	return a[ math.random(#a) ]
end

function M.rayleigh_rand(sigma)
	return sigma * math.sqrt( -2 * math.log(1-math.random()) )
end


return M

--[=[

=pod

=head1 NAME

randomdist.lua - 
a few simple procedures for generating random numbers.

=head1 SYNOPSIS

 local R = require 'randomdist'

 grand1 = R.new_grand(10,3)
 grand2 = R.new_grand(100,3)
 for i = 1,20 do print( grand1(), grand2() ) end

 gue_irand1 = R.new_gue_irand(4)
 gue_irand2 = R.new_gue_irand(20)
 for i = 1,20 do print( gue_irand1(), gue_irand2() ) end

 for i = 1,20 do print(R.rayleigh_rand(3)) end

 a = {'cold', 'cool', 'warm', 'hot'}
 for i = 1,20 do print(R.randomget(a)) end

=head1 DESCRIPTION

This module implements in Lua a few simple functions
for generating random numbers according to various distributions.

=head1 FUNCTIONS

=over 3

=item I<new_grand( mean, stddev)>

This function returns a closure, which is a function which you
can then call to return a Gaussian (or Normal) Random distribution of numbers
with the given I<mean> and I<standard deviation>.

It keeps some internal local state, but because it is a closure, 
you may run different Gaussian Random generators simultaneously,
for example with different means and standard-deviations,
without them interfering with each other.

It uses the algorithm given by Erik Carter in
http://www.design.caltech.edu/erik/Misc/Gaussian.html

This algorithm generates results in pairs, but returns them one by one.
Therefore if you are using I<math.randomseed> to reset the random-number
generator to a known state, and your code happens to make an odd number
of calls to your closure, and you want your program to run consistently,
then you should call your closure (eg: I<grand1>) with the
argument 'reset' each time you call I<math.randomseed>. Eg:

 grand1 = R.new_grand(10,3)
 ... grand1() ... etc ...
 math.randomseed(244823040) ; grand1('reset')

=item I<new_gue_irand( average )>

This function returns a closure, which is a function which you can then
call to return a Gaussian-Random-Ensemble distribution of integers.

The Gaussian Unitary Ensemble models Hamiltonians lacking
time-reversal symmetry.
Considering a hermitian matrix with gaussian-random values;
from the ordered sequence of eigenvalues,
one defines the normalized spacings

 s = (\lambda_{n+1}-\lambda_n) / <s>

where <s> = is the mean spacing.
The probability distribution of spacings is approximately given by

  p_2(s) = (32 / pi^2) * s^2 * e^((-4/pi) * s^2)

These numerical constants are such that p_2 (s) is normalized:
and the mean spacing is 1.

  \int_0^\infty ds p_2(s) = 1 
  \int_0^\infty ds s p_2(s) = 1

Montgomery's pair correlation conjecture is a conjecture made by Hugh
Montgomery (1973) that the pair correlation between pairs of zeros of
the Riemann zeta function (normalized to have unit average spacing) is:

  1 - ({sin(pi u)}/{pi u}})^2 + \delta(u)

which, as Freeman Dyson pointed out to him, is the same as the pair
correlation function of random Hermitian matrices.

=item I<rayleigh_rand( sigma )>

This function returns a random number according to the Rayleigh Distribution,
which is a continuous probability distribution for positive-valued
random variables.  It occurs, for example, when random complex numbers
whose real and imaginary components are independent Gaussian distributions
with equal variance and zero mean, in which case,
the absolute value of the complex number is Rayleigh-distributed.

 f(x; sigma) = x exp(-x^2 / 2*sigma^2) / sigma^2      for x>=0

The algorithm contains no internal state,
hence I<rayleigh_rand> directly returns a number.

=item I<randomget( an_array )>

This example gets a random element from the given array.
For example, the following executes one of the given procedures at random:

  f = {bassclef, trebleclef, sharp, natural} randomget()
  f()

=back

=head1 DOWNLOAD

This module is available at
http://www.pjb.com.au/comp/lua/randomdist.html

=head1 AUTHOR

Peter J Billam, http://www.pjb.com.au/comp/contact.html

=head1 SEE ALSO

 https://en.wikipedia.org/wiki/Normal_distribution
 http://www.design.caltech.edu/erik/Misc/Gaussian.html
 https://en.wikipedia.org/wiki/Random_matrix#Gaussian_ensembles
 https://en.wikipedia.org/wiki/Random_matrix#Distribution_of_level_spacings
 https://en.wikipedia.org/wiki/Montgomery%27s_pair_correlation_conjecture
 https://en.wikipedia.org/wiki/Radial_distribution_function
 https://en.wikipedia.org/wiki/Pair_distribution_function
 https://en.wikipedia.org/wiki/Rayleigh_distribution
 https://luarocks.org/modules/luarocks/lrandom
 http://www.pjb.com.au/comp/randomdist.html
 http://www.pjb.com.au/comp/index.html

 Montgomery, Hugh L. (1973), "The pair correlation of zeros of the zeta
 function", Analytic number theory, Proc. Sympos. Pure Math., XXIV,
 Providence, R.I.: American Mathematical Society, pp. 181-193, MR 0337821

 Odlyzko, A. M. (1987), "On the distribution of spacings between zeros
 of the zeta function", Mathematics of Computation, American Mathematical
 Society, 48 (177): 273-308, ISSN 0025-5718, JSTOR 2007890, MR 866115,
 doi:10.2307/2007890

 "Prime Obsession", by John Derbyshire, p.288

=cut

]=]

