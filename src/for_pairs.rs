use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

pub trait ForPairs<E> {
    fn par_for_pairs<T: Send>(
        &mut self,
        map: impl Fn(&E, &E) -> Option<(T, T)> + Sync,
        apply: impl FnMut(&mut E, T),
    );
}

impl<E: Sync> ForPairs<E> for Vec<E>
where
    Vec<E>: IntoParallelIterator<Item = E>,
    <Vec<E> as IntoParallelIterator>::Iter: ParallelIterator + IndexedParallelIterator,
{
    fn par_for_pairs<T: Send>(
        &mut self,
        map: impl Fn(&E, &E) -> Option<(T, T)> + Sync,
        mut apply: impl FnMut(&mut E, T),
    ) {
        let v = self
            .par_iter()
            .enumerate()
            .map(|(i, o0)| {
                self[i + 1..]
                    .iter()
                    .enumerate()
                    .filter_map(|(j, o1)| map(o0, o1).map(|(a, b)| ((i, a), (i + 1 + j, b))))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        v.into_iter()
            .flat_map(|v| v.into_iter())
            .for_each(|((i, a), (j, b))| {
                apply(&mut self[i], a);
                apply(&mut self[j], b);
            });
    }
}
