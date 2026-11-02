#pragma once
#include <cstddef>
#include <memory>

namespace videohelper {
// Remembers that an object inside a plan publication already compared equal. A publication's
// plans are never written once it is shared, so the same publication and the same address hold
// the same bytes. A weak owner cannot be fooled by a new publication at a freed address.
// A null publication remembers nothing, and the caller compares in full.
class PublishedPlanMemo final {
public:
  bool holds(const std::shared_ptr<const void>& publication, const void* object) const noexcept {
    return publication && object && object==object_ && publication_.lock()==publication;
  }
  void remember(const std::shared_ptr<const void>& publication, const void* object) noexcept {
    publication_=publication;
    object_=publication ? object : nullptr;
  }
  void clear() noexcept { publication_.reset(); object_=nullptr; }
private:
  std::weak_ptr<const void> publication_;
  const void* object_=nullptr;
};
} // namespace videohelper
