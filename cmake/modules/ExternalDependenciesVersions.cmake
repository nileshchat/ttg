# for each dependency track both current and previous id (the variable for the latter must contain PREVIOUS)
# to be able to auto-update them

# need Boost.CallableTraits (header only, part of Boost 1.66 released in Dec 2017) for wrap.h to work
set(TTG_TRACKED_BOOST_VERSION 1.66)
set(TTG_TRACKED_CATCH2_VERSION 2.13.1)
set(TTG_TRACKED_CEREAL_VERSION 1.3.0)
set(TTG_TRACKED_MADNESS_TAG 687ac4f5308c82de04ea0f803f57417fa92713d5)
set(TTG_TRACKED_PARSEC_TAG cb9322046c1f856fbca5804a9f96b83bbe2e676b)
set(TTG_TRACKED_BTAS_TAG bef94e50416bcfbe2bf868ece52a021d7459e7c0)
